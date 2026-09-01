// =============================================================================
//  Smart City Audio Classifier - Raspberry Pi Pico (RP2040)
//  FreeRTOS + TensorFlow Lite Micro
//
//  PRINCIPIO DI PROGETTO: il sistema si diagnostica da solo.
//  Ogni fase dell'avvio viene registrata in un registro che sopravvive al
//  reset. Un watchdog hardware riavvia il chip se una fase non si completa,
//  e al riavvio il sistema STAMPA in quale fase era rimasto bloccato.
//  Non serve guardare il LED ne' collegarsi al momento giusto: il messaggio
//  viene ristampato a ogni riavvio, quindi lo si legge quando si vuole.
//
//  DA FARE IN CMakeLists.txt:
//      target_link_libraries(... hardware_dma hardware_watchdog ...)
// =============================================================================
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/adc.h"
#include "hardware/regs/adc.h"
#include "hardware/dma.h"
#include "hardware/clocks.h"
#include "hardware/watchdog.h"
#include "hardware/structs/watchdog.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const unsigned char g_model[];
extern const unsigned int  g_model_len;

#include "mel_filters_sparse.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// =============================================================================
//  CONFIGURAZIONE
// =============================================================================
#define SAMPLE_RATE   16000
// L'ADC campiona a 48 kHz e un filtro digitale decima a 16 kHz. Cosi' la banda
// 8-24 kHz viene rimossa nel digitale invece di ripiegarsi nello spettrogramma.
#define ADC_OVERSAMPLE 3
#define ADC_RATE      (SAMPLE_RATE * ADC_OVERSAMPLE)   // 48000
#define FIR_TAPS      31
#define N_FFT         512
#define ADC_BLOCK     (N_FFT * ADC_OVERSAMPLE)         // 1536 campioni grezzi
#define HOP_LENGTH    256                              // meta' di N_FFT
#define FRAMES_PER_BLOCK (N_FFT / HOP_LENGTH)           // 2 colonne per blocco
#define N_MELS        MEL_N_BANDS       // 20
#define N_BINS        (N_FFT / 2 + 1)   // 257
#define FRAMES        61                // finestra da 1.0 s con center=False:
                                        // 1 + (16000-512)/256 = 61 colonne
#define HOP_FRAMES    31                // inferenza ogni 31 frame (~0.50 s)

#define MIN_DB        (-80.0f)
#define MAX_DB        (20.0f)
#define PEAK_NORMALIZE 1                // coerente con extract_features.py

// Sotto questa ampiezza di picco la finestra e' solo rumore di fondo.
// La normalizzazione al picco - necessaria per allinearsi al dataset, dove
// ogni clip e' normalizzata - amplificherebbe quel rumore fino a fondo scala,
// producendo uno spettrogramma a larga banda che somiglia a un transiente.
// E' il motivo per cui il sistema dichiarava "Spari" nei momenti di silenzio.
#define SOGLIA_SILENZIO  0.02f          // ~40 conteggi su 2048, cioe' il 2%
// Sotto questa confidenza la predizione e' dichiarata incerta anziche' forzata.
// Sul Pico le probabilita' sono piu' compresse che sul PC, quindi la soglia
// va tenuta piu' bassa di quanto si userebbe in fase di sviluppo.
#define SOGLIA_CONFIDENZA 0.45f

// [!] VERIFICA QUESTI DUE: canale 0=GPIO26, 1=GPIO27, 2=GPIO28
#define ADC_GPIO      27
#define ADC_CHANNEL   1

// Passi della finestra temporale: deve coincidere con make_rnn_dataset.py,
// dove vale (durata - FINESTRA) / PASSO + 1 = (8 - 1) / 0.5 + 1 = 15.
// ---- Macchina a stati deterministica -----------------------------------
// Affianca il modello temporale invece di sostituirlo: la rete e' brava sugli
// scenari continui (voce, emergenza), le regole sugli eventi puntuali (sparo,
// incidente). I tempi sono in CICLI di inferenza, uno ogni ~0,5 s.
#define ALPHA_LIV        0.40f   // media esponenziale, costante di tempo ~1 s
#define SOGLIA_EVENTO    0.50f   // oltre questa, la classe impulsiva "e' successa"
// Incidente e Vetri raramente salgono: un impatto forte viene spesso letto
// come Spari o Fuochi, e i vetri restano sepolti nel resto del rumore.
// Contarli solo sopra 0,50 significa non contarli quasi mai.
#define SOGLIA_DEBOLE    0.25f
// Sirene_e_Urla riconosce bene anche lo STRIDIO DI GOMME, che nella dinamica
// di un incidente precede l'impatto. E' l'indizio piu' affidabile che abbiamo.
#define SOGLIA_STRIDIO   0.40f
#define FIN_INCIDENTE   12       // 6 s: finestra in cui gli indizi devono cadere
// Vetri come prova AUTONOMA, fuori da un contesto di incidente. Piu' alta di
// SOGLIA_DEBOLE perche' qui e' l'unico indizio: un indizio debole vale in
// compagnia, non da solo.
#define SOGLIA_VANDALISMO 0.35f
#define SOGLIA_LIVELLO   0.30f   // oltre questa, la classe continua "e' presente"
#define REFRATTARIO      2       // 1 s: uno stesso scoppio non conta due volte
#define ISOLAMENTO       8       // 4 s senza altri scoppi -> sparo isolato
#define FIN_FUOCHI      20       // 10 s
#define MIN_FUOCHI       3       // scoppi in FIN_FUOCHI per dire fuochi d'artificio
#define SIRENE_CONTINUE  6       // 3 s di sirene ininterrotte -> emergenza
// Contesto immediatamente precedente all'urto: veicoli OPPURE persone.
// Un secondo, non dieci: quello che conta e' che la scena fosse viva un
// istante prima dell'impatto, non che ci sia passata una macchina prima.
#define MEM_CONTESTO     2       // 1 s
// (FIN_IMPATTO rimosso: la finestra degli indizi e' ora FIN_INCIDENTE)
#define DURATA_MIN      10       // 5 s: uno stato non puo' essere abbassato prima

// ---- Registrazione di prova --------------------------------------------
// Buffer CIRCOLARE sempre in scrittura: quando il rilevatore si accorge del
// picco, quei campioni sono gia' in memoria da 32 ms. La latenza toglie tempo
// alla coda, non all'attacco. A 8 bit per stare nei 32 KB gia' allocati.
// Un solo buffer circolare per tutto: clip di allerta e registrazioni su
// comando. Prima ce n'erano due (32 KB a 16 bit per il dataset, 32 KB a 8 bit
// per le clip); rimuovendo il modello temporale e unificandoli si arriva a
// 4 secondi invece di 2.
#define CLIP_SAMPLES  64000      // 4 s a 16 kHz, 8 bit -> 64 KB
#define CLIP_PRE      32000      // 2 s riversati come clip di allerta
#define REC_CAMPIONI  16000      // 1 s riversato dal comando 'r' (dataset)
#define TRANS_FATTORE   4.0f     // energia del blocco / media -> transiente
// 2 s a 8 bit sono 32.000 byte, cioe' ~42.700 caratteri base64: a 115200 baud
// circa 3,7 s di trasmissione. Il tempo morto deve stare sopra quel valore con
// margine, e 10 s ne lasciano quasi il triplo. Accorciare la clip da 3,5 a 2 s
// quasi dimezza la finestra in cui un secondo evento non verrebbe registrato.
#define TEMPO_MORTO     20       // cicli (~10 s) fra due riversamenti
#define CAND_VALIDO      4       // cicli entro cui la CNN deve confermare

#define LED_PIN       PICO_DEFAULT_LED_PIN

// ---- Pulsante di dimostrazione -----------------------------------------
// Collegato fra GPIO15 (pin fisico 20) e GND, con pull-up interno: a riposo
// legge 1, premuto legge 0. Serve a far partire dal PC la riproduzione di un
// campione, per mostrare il riconoscimento durante una presentazione.
#define PULSANTE_GPIO   15
#define PULSANTE_STABILI 3       // letture consecutive a 32 ms = ~96 ms
#define PULSANTE_MORTO  30       // ~1 s prima di accettare una nuova pressione
// Il clic meccanico arriva al microfono e viene classificato come scoppio.
// Non e' un falso positivo da tarare con le soglie: e' un evento che il
// sistema stesso ha causato e di cui conosce l'istante esatto, quindi si
// ignorano le finestre che lo contengono. Il clic dura pochi millisecondi ma
// la finestra di inferenza e' lunga 1 s e scorre di 0,5: lo stesso clic
// compare in due finestre consecutive, e servono almeno quelle.
#define PULSANTE_CIECO   3       // cicli di inferenza ignorati (~1,5 s)
// Registrazione su richiesta: 1 secondo a 16 kHz = 32 KB di RAM.
// Serve a raccogliere campioni CON LA CATENA REALE (microfono, ADC, filtro
// di decimazione) invece che con i file del dataset. E' l'unico modo per
// misurare l'accuratezza vera e per fare fine-tuning su dati propri.

// 40 KB: il buffer di registrazione ne occupa 32. Controlla la riga
// "arena usata X/Y" all'avvio; con WIDTH=2 potrebbe servirne di piu'.
#define ARENA_SIZE    (40 * 1024)
#define WATCHDOG_MS   4000              // massimo consentito ~8300
#define ML_TIMEOUT_MS 5000              // oltre questo Task_ML e' considerato bloccato

// =============================================================================
//  TRACCIAMENTO DELLE FASI (sopravvive al reset del watchdog)
//  scratch[0] del watchdog non viene azzerato dal reset.
// =============================================================================
#define STAGE_MAGIC   0x5AFE0000u

// scratch[0] = fase di main + Task_ML     scratch[4] = fase di Task_DSP
// scratch[1] = codice di fault             scratch[5] = blocchi audio elaborati
// scratch[2] = PC del fault                scratch[6] = inferenze completate
// scratch[3] = LR del fault
//
// I due percorsi hanno registri DISTINTI: se condividessero lo stesso, il task
// piu' avanzato cancellerebbe l'informazione dell'altro. E' esattamente
// l'errore che ha reso illeggibile la diagnosi precedente.

enum Stage : uint32_t {
    ST_BOOT = 1,
    ST_STDIO,
    ST_ML_START,
    ST_TFLM_GETMODEL,
    ST_TFLM_ALLOCATE,
    ST_WARMUP_ENTER,
    ST_WARMUP_DONE,
    ST_WAIT_WINDOW,
    ST_INVOKE_ENTER,
    ST_RUNNING
};

enum DspStage : uint32_t {
    DS_START = 1,
    DS_DSP_INIT,
    DS_ADC_DMA,
    DS_FIRST_BLOCK,
    DS_WINDOW_SENT
};

static const char* stage_name(uint32_t s) {
    switch (s) {
        case ST_BOOT:          return "avvio";
        case ST_STDIO:         return "seriale USB pronta";
        case ST_ML_START:      return "Task_ML avviato";
        case ST_TFLM_GETMODEL: return "lettura del modello";
        case ST_TFLM_ALLOCATE: return "AllocateTensors";
        case ST_WARMUP_ENTER:  return "DENTRO Invoke() a vuoto <<<";
        case ST_WARMUP_DONE:   return "Invoke() a vuoto completato";
        case ST_WAIT_WINDOW:   return "in attesa di una finestra";
        case ST_INVOKE_ENTER:  return "DENTRO Invoke() con dati reali <<<";
        case ST_RUNNING:       return "regime normale";
        default:               return "sconosciuta";
    }
}

static const char* dsp_stage_name(uint32_t s) {
    switch (s) {
        case DS_START:       return "Task_DSP avviato";
        case DS_DSP_INIT:    return "tabelle DSP pronte";
        case DS_ADC_DMA:     return "ADC + DMA avviati";
        case DS_FIRST_BLOCK: return "primo blocco audio ricevuto";
        case DS_WINDOW_SENT: return "finestra consegnata a Task_ML";
        default:             return "sconosciuta";
    }
}

static inline void stage_set(uint32_t s) { watchdog_hw->scratch[0] = STAGE_MAGIC | s; }
static inline uint32_t stage_get(void) {
    uint32_t v = watchdog_hw->scratch[0];
    return ((v & 0xFFFF0000u) == STAGE_MAGIC) ? (v & 0xFFFFu) : 0;
}
static inline void dsp_stage_set(uint32_t s) { watchdog_hw->scratch[4] = STAGE_MAGIC | s; }
static inline uint32_t dsp_stage_get(void) {
    uint32_t v = watchdog_hw->scratch[4];
    return ((v & 0xFFFF0000u) == STAGE_MAGIC) ? (v & 0xFFFFu) : 0;
}

// =============================================================================
//  STAMPA DAVVERO NON BLOCCANTE
//
//  Lezione imparata a caro prezzo: stdio_usb PUO' bloccare indefinitamente
//  anche quando stdio_usb_connected() dice true, perche' se il buffer CDC si
//  riempie e il servizio USB non gira, vprintf resta in attesa per sempre.
//  Un task bloccato li' dentro sembra bloccato nell'algoritmo che stava
//  eseguendo, e manda fuori strada ogni diagnosi.
//
//  Regola: NESSUN task scrive su stdout. say() formatta la riga in un buffer
//  circolare e ne accoda l'indice. Un solo task, il meno prioritario, svuota
//  la coda e stampa. Se quello si blocca si perdono messaggi, ma DSP e
//  inferenza continuano e il watchdog continua a essere alimentato.
// =============================================================================
#define LOG_LINES 16
#define LOG_LEN   320

static char          g_log_buf[LOG_LINES][LOG_LEN];
static uint8_t       g_log_w = 0;
static QueueHandle_t g_log_q = NULL;

static void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (g_log_q == NULL) {
        // Prima dello scheduler non c'e' concorrenza: stampa diretta.
        vprintf(fmt, ap);
        va_end(ap);
        fflush(stdout);
        return;
    }
    taskENTER_CRITICAL();
    uint8_t idx = g_log_w;
    g_log_w = (uint8_t)((g_log_w + 1) % LOG_LINES);
    taskEXIT_CRITICAL();

    vsnprintf(g_log_buf[idx], LOG_LEN, fmt, ap);
    va_end(ap);
    xQueueSend(g_log_q, &idx, 0);       // non bloccante: se piena, la riga si perde
}

// -----------------------------------------------------------------------------
//  Stato della registrazione. Un solo scrittore (Task_DSP) e un solo lettore
//  (Task_Serial), sincronizzati dallo stato: niente lock.
// -----------------------------------------------------------------------------
// ---- Buffer circolare, scritto sempre da Task_DSP ------------------------
static volatile bool     g_rec_richiesta = false;   // comando 'r' ricevuto
static int8_t            g_clip[CLIP_SAMPLES];
static volatile uint32_t g_clip_w = 0;
static volatile bool     g_clip_congelato = false;   // ferma la scrittura
static volatile bool     g_clip_pronta = false;      // da riversare
static volatile bool     g_candidato = false;        // transiente rilevato
static volatile bool     g_demo_richiesta = false;   // pulsante premuto
static volatile uint32_t g_demo_n = 0;               // quante pressioni
static volatile bool     g_arma_cieco = false;       // chiedi a Task_ML di ignorare
// Rispecchia lo stato "sono nella finestra cieca" per Task_DSP: senza questo
// il rilevatore rapido restava attivo durante il clic del pulsante, che e'
// un vero picco di energia, e poteva marcare un candidato con quel suono.
// Se in quel momento lo stato era ancora pericoloso per isteresi (fino a 5 s
// dopo un evento vero), il candidato del clic soddisfaceva la condizione di
// innesco e la clip catturava la pressione del pulsante invece dell'evento.
static volatile bool     g_pulsante_cieco = false;
static volatile int      g_pin_cambiato = -1;        // livello da segnalare, -1 = nulla
static volatile uint32_t g_clip_perse = 0;           // inneschi rifiutati

// =============================================================================
//  ACQUISIZIONE: ADC free-running a 16 kHz -> DMA ping-pong su due buffer
// =============================================================================
static uint16_t          g_adc_buf[2][ADC_BLOCK];
static volatile int      g_adc_active = 0;
static volatile uint32_t g_adc_overruns = 0;
static SemaphoreHandle_t g_sem_block;
static int               g_dma_ch = -1;

static void __isr dma_block_handler(void) {
    dma_hw->ints0 = 1u << g_dma_ch;
    g_adc_active ^= 1;
    dma_channel_set_write_addr(g_dma_ch, g_adc_buf[g_adc_active], true);

    BaseType_t hpw = pdFALSE;
    if (xSemaphoreGiveFromISR(g_sem_block, &hpw) != pdTRUE) g_adc_overruns++;
    portYIELD_FROM_ISR(hpw);
}

static void adc_dma_start(void) {
    adc_init();
    adc_gpio_init(ADC_GPIO);
    adc_select_input(ADC_CHANNEL);
    adc_fifo_setup(true, true, 1, false, false);
    adc_fifo_drain();
    adc_set_clkdiv((float)(48000000.0 / ADC_RATE) - 1.0f);      // 999 -> 48 kHz

    g_dma_ch = dma_claim_unused_channel(true);
    dma_channel_config c = dma_channel_get_default_config(g_dma_ch);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_16);
    channel_config_set_read_increment(&c, false);
    channel_config_set_write_increment(&c, true);
    channel_config_set_dreq(&c, DREQ_ADC);
    dma_channel_configure(g_dma_ch, &c, g_adc_buf[0], &adc_hw->fifo, ADC_BLOCK, false);

    dma_channel_set_irq0_enabled(g_dma_ch, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_block_handler);
    irq_set_priority(DMA_IRQ_0, 0xC0);          // priorita' bassa: sicura con FreeRTOS
    irq_set_enabled(DMA_IRQ_0, true);

    dma_channel_start(g_dma_ch);
    adc_run(true);
}

// =============================================================================
//  DSP: FFT reale a 512 punti tramite FFT complessa a 256 punti + unpacking.
//  Verificata contro numpy.fft.rfft: errore relativo massimo 5.8e-8.
// =============================================================================
#define NH (N_FFT / 2)

static float   g_tw_re[NH + 1], g_tw_im[NH + 1];
static uint8_t g_brev[NH];
static float   g_hann[N_FFT];

static void dsp_init(void) {
    for (int k = 0; k <= NH; k++) {
        float a = -2.0f * M_PI * (float)k / (float)N_FFT;
        g_tw_re[k] = cosf(a);
        g_tw_im[k] = sinf(a);
    }
    for (int i = 0; i < NH; i++) {
        int r = 0;
        for (int b = 0; b < 8; b++) if (i & (1 << b)) r |= 1 << (7 - b);
        g_brev[i] = (uint8_t)r;
    }
    // Hann PERIODICA (denominatore N), come scipy/librosa
    for (int i = 0; i < N_FFT; i++)
        g_hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)N_FFT));
}

static void fft256(float* zr, float* zi) {
    for (int i = 0; i < NH; i++) {
        int j = g_brev[i];
        if (i < j) {
            float t = zr[i]; zr[i] = zr[j]; zr[j] = t;
            t = zi[i]; zi[i] = zi[j]; zi[j] = t;
        }
    }
    for (int len = 2; len <= NH; len <<= 1) {
        int half = len >> 1, stride = (NH / len) * 2;
        for (int i = 0; i < NH; i += len) {
            int t = 0;
            for (int k = 0; k < half; k++) {
                float wr = g_tw_re[t], wi = g_tw_im[t];
                float xr = zr[i + k + half], xi = zi[i + k + half];
                float vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
                float ur = zr[i + k], ui = zi[i + k];
                zr[i + k]        = ur + vr;  zi[i + k]        = ui + vi;
                zr[i + k + half] = ur - vr;  zi[i + k + half] = ui - vi;
                t += stride;
            }
        }
    }
}

static void real_power_spectrum(const float* x, float* power) {
    static float zr[NH], zi[NH];
    for (int k = 0; k < NH; k++) { zr[k] = x[2 * k]; zi[k] = x[2 * k + 1]; }
    fft256(zr, zi);
    for (int k = 0; k <= NH; k++) {
        int ik = (k == NH) ? 0 : k;
        int in = (NH - k) & (NH - 1);
        float a = zr[ik], b = zi[ik], c = zr[in], d = zi[in];
        float fe_r =  0.5f * (a + c), fe_i =  0.5f * (b - d);
        float fo_r =  0.5f * (b + d), fo_i = -0.5f * (a - c);
        float wr = g_tw_re[k], wi = g_tw_im[k];
        float xr = fe_r + (wr * fo_r - wi * fo_i);
        float xi = fe_i + (wr * fo_i + wi * fo_r);
        power[k] = xr * xr + xi * xi;
    }
}

// =============================================================================
//  Finestra scorrevole di colonne log-mel
// =============================================================================
static float g_mel_ring[FRAMES][N_MELS];
static float g_peak_ring[FRAMES];
static int   g_ring_w = 0;
static int   g_frames_filled = 0;

static float             g_window_db[N_MELS * FRAMES];
static volatile bool     g_finestra_debole = false;
static volatile bool     g_ml_busy = false;
static volatile uint32_t g_dropped = 0;
static SemaphoreHandle_t g_sem_window;

// Filtro FIR passa-basso a 31 coefficienti in Q15: taglio a 7 kHz su fs=48 kHz.
// Elimina la banda che, decimando a 16 kHz, si ripiegherebbe dentro lo
// spettrogramma. Attenuazione: -60 dB a 12 kHz, -54 dB a 16 kHz.
static const int16_t FIR_Q15[FIR_TAPS] = {
      51,     17,    -58,   -146,   -134,     84,    426,    555,
     114,   -838,  -1592,  -1105,   1213,   4834,   8186,   9554,
    8186,   4834,   1213,  -1105,  -1592,   -838,    114,    555,
     426,     84,   -134,   -146,    -58,     17,     51,
};

// Livelli dell'ultimo blocco, per la taratura del trimmer del microfono
static volatile uint16_t g_lvl_min = 0, g_lvl_max = 0;
static volatile float    g_lvl_dc = 0.0f, g_lvl_rms = 0.0f;

// Calcola UNA colonna log-mel da 512 campioni a 16 kHz gia' centrati sullo zero.
static void emit_frame(const int16_t* dec) {
    static float frame[N_FFT];
    static float power[N_BINS];

    float peak = 0.0f;
    for (int i = 0; i < N_FFT; i++) {
        float v = (float)dec[i] * (1.0f / 2048.0f);
        float a = fabsf(v);
        if (a > peak) peak = a;
        frame[i] = v * g_hann[i];
    }

    real_power_spectrum(frame, power);

    int w = g_ring_w;
    g_peak_ring[w] = peak;

    // Mel sparso: solo i pesi non nulli di ogni filtro triangolare
    for (int m = 0; m < N_MELS; m++) {
        const MelBand* b  = &MEL_BANDS[m];
        const float*   wt = &MEL_WEIGHTS[b->w_offset];
        const float*   ps = &power[b->start_bin];
        float e = 0.0f;
        for (int i = 0; i < b->n_bins; i++) e += ps[i] * wt[i];
        g_mel_ring[w][m] = 10.0f * log10f(fmaxf(e, 1e-10f));
    }

    g_ring_w = (w + 1) % FRAMES;
    if (g_frames_filled < FRAMES) g_frames_filled++;
}

// Un blocco DMA porta 1536 campioni a 48 kHz = 512 a 16 kHz = 32 ms di audio.
// Con hop 256 questi 512 campioni completano DUE colonne dello spettrogramma,
// entrambe sovrapposte al blocco precedente: serve quindi una coda di 256
// campioni riportata da un blocco al successivo.
static void process_block(const uint16_t* raw) {
    static int16_t hist[FIR_TAPS - 1];                    // coda del FIR
    static int16_t x[FIR_TAPS - 1 + ADC_BLOCK];
    static int16_t win[HOP_LENGTH + N_FFT];               // coda + blocco nuovo
    static bool    primed = false;

    // 1) DC misurato sul blocco, piu' min/max per il misuratore di livello
    uint32_t sum = 0;
    uint16_t mn = 4095, mx = 0;
    for (int i = 0; i < ADC_BLOCK; i++) {
        uint16_t v = raw[i];
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    int32_t dc = (int32_t)(sum / ADC_BLOCK);
    g_lvl_min = mn; g_lvl_max = mx; g_lvl_dc = (float)dc;

    // 2) coda del FIR + campioni centrati sullo zero
    for (int i = 0; i < FIR_TAPS - 1; i++) x[i] = hist[i];
    for (int i = 0; i < ADC_BLOCK; i++)
        x[FIR_TAPS - 1 + i] = (int16_t)((int32_t)raw[i] - dc);
    for (int i = 0; i < FIR_TAPS - 1; i++) hist[i] = x[ADC_BLOCK + i];

    // 3) FIR e decimazione 3:1 -> 512 campioni a 16 kHz, scritti dopo la coda
    float sq = 0.0f;
    for (int j = 0; j < N_FFT; j++) {
        int32_t acc = 0;
        const int16_t* pz = &x[j * ADC_OVERSAMPLE];
        for (int k = 0; k < FIR_TAPS; k++) acc += (int32_t)FIR_Q15[k] * (int32_t)pz[k];
        int16_t d = (int16_t)(acc >> 15);
        win[HOP_LENGTH + j] = d;
        sq += (float)d * (float)d;

        // Buffer circolare a 8 bit, sempre in scrittura salvo durante il
        // riversamento. Congelare evita che il WAV risulti un miscuglio di
        // due momenti diversi.
        if (!g_clip_congelato) {
            int32_t c8 = d >> 4;                     // +-2048 -> +-128
            if (c8 < -128) c8 = -128;
            if (c8 >  127) c8 =  127;
            g_clip[g_clip_w] = (int8_t)c8;
            g_clip_w = (g_clip_w + 1) % CLIP_SAMPLES;
        }
    }
    g_lvl_rms = sqrtf(sq / (float)N_FFT);

    // Rilevatore rapido di transiente: gira a 32 ms, non a 500 come la CNN.
    // Non classifica nulla, distingue solo "e' successo qualcosa". Serve a
    // marcare un CANDIDATO; la conferma arriva dopo dall'inferenza.
    // Silenziato durante la finestra cieca del pulsante: il clic e' un vero
    // transiente energetico e non deve poter armare l'innesco della clip.
    static float media_energia = 0.0f;
    float e = g_lvl_rms;
    if (!g_pulsante_cieco && media_energia > 1.0f && e > TRANS_FATTORE * media_energia)
        g_candidato = true;
    media_energia = 0.05f * e + 0.95f * media_energia;

    // 4) Due colonne: la prima a cavallo fra blocco precedente e attuale,
    //    la seconda interamente dentro quello attuale.
    //    Il primo blocco in assoluto non ha coda valida: se ne emette una sola.
    if (primed) emit_frame(&win[0]);
    emit_frame(&win[HOP_LENGTH]);
    primed = true;

    // 5) Gli ultimi HOP_LENGTH campioni diventano la coda del prossimo blocco
    for (int i = 0; i < HOP_LENGTH; i++) win[i] = win[N_FFT + i];
}

static void build_window(void) {
    float gain_db = 0.0f;
#if PEAK_NORMALIZE
    // Normalizzare l'audio a picco 1.0 equivale a sottrarre 20*log10(picco) dB
    // a tutte le bande: esatto, e si puo' fare qui dopo la log.
    float peak = 1e-6f;
    for (int f = 0; f < FRAMES; f++) if (g_peak_ring[f] > peak) peak = g_peak_ring[f];

    if (peak < SOGLIA_SILENZIO) {
        // Finestra troppo debole: si applica un guadagno FISSO invece di
        // normalizzare, cosi' il rumore non viene portato a fondo scala.
        g_finestra_debole = true;
        gain_db = 20.0f * log10f(SOGLIA_SILENZIO);
    } else {
        g_finestra_debole = false;
        gain_db = 20.0f * log10f(peak);
    }
#endif
    int start = g_ring_w;
    for (int t = 0; t < FRAMES; t++) {
        const float* col = g_mel_ring[(start + t) % FRAMES];
        // Layout del tensore (1, N_MELS, FRAMES, 1) -> indice m*FRAMES + t
        for (int m = 0; m < N_MELS; m++)
            g_window_db[m * FRAMES + t] = col[m] - gain_db;
    }
}

// =============================================================================
//  TASK 1 - DSP (priorita' 3). Un blocco ogni 32 ms.
//  E' anche il guardiano del watchdog: smette di alimentarlo se Task_ML
//  non da' segni di vita entro ML_TIMEOUT_MS, provocando il riavvio.
// =============================================================================
static volatile uint32_t   g_dsp_us_last = 0, g_dsp_us_max = 0;
static volatile TickType_t g_ml_deadline = 0;    // 0 = nessuna scadenza attiva

static void Task_DSP(void* pv) {
    (void)pv;
    dsp_stage_set(DS_START);
    dsp_init();
    dsp_stage_set(DS_DSP_INIT);
    adc_dma_start();
    dsp_stage_set(DS_ADC_DMA);

    int      since_last = 0;
    uint32_t blocks = 0;

    for (;;) {
        if (xSemaphoreTake(g_sem_block, pdMS_TO_TICKS(1000)) != pdTRUE) {
            say("ATTENZIONE: nessun blocco dal DMA. busy=%d rimasti=%u adc_cs=0x%08x\n",
                (int)dma_channel_is_busy(g_dma_ch),
                (unsigned)dma_hw->ch[g_dma_ch].transfer_count,
                (unsigned)adc_hw->cs);
            continue;                            // watchdog non alimentato -> riavvio
        }

        uint32_t t0 = time_us_32();
        process_block(g_adc_buf[g_adc_active ^ 1]);
        uint32_t dt = time_us_32() - t0;
        g_dsp_us_last = dt;
        if (dt > g_dsp_us_max) g_dsp_us_max = dt;

        // ---- pulsante di dimostrazione ----
        // Letto QUI e non in Task_Serial: questo task ha periodo fisso di
        // 32 ms e non si blocca mai, quindi il campionamento e' deterministico
        // e le tre letture consecutive fanno da antirimbalzo senza timer.
        // Task_Serial invece puo' restare occupato per secondi durante il
        // riversamento di una clip, e il pulsante non risponderebbe.
        {
            static uint8_t  bassi = 0, alti = 0;
            static bool     gia_scattato = false;
            static uint32_t ultimo_demo = 0;

            // Monitor: segnala ogni cambio di livello. Serve a capire se il
            // pulsante commuta davvero. A riposo deve leggere 1 e premuto 0;
            // se non cambia mai, il contatto e' chiuso o aperto in permanenza
            // e nessuna modifica al codice puo' farlo funzionare.
            static int ultimo_liv = -1;
            int liv = gpio_get(PULSANTE_GPIO);
            if (liv != ultimo_liv) { ultimo_liv = liv; g_pin_cambiato = liv; }

            // Antirimbalzo SIMMETRICO. Prima bastava una sola lettura alta
            // per riarmare lo scatto: con un contatto rumoroso, o con un filo
            // libero che fa da antenna sul pull-up interno (~50 kohm, debole),
            // un singolo disturbo riabilitava e il livello basso successivo
            // produceva un altro scatto. Da qui gli scatti a raffica.
            // Ora servono N letture consecutive in ENTRAMBI i versi.
            if (liv == 0) {                              // premuto (pull-up)
                if (bassi < PULSANTE_STABILI) bassi++;
                alti = 0;
                if (bassi >= PULSANTE_STABILI && !gia_scattato &&
                    (blocks - ultimo_demo) > PULSANTE_MORTO) {
                    gia_scattato = true;                 // uno scatto per pressione
                    ultimo_demo = blocks;
                    g_demo_n++;
                    g_demo_richiesta = true;
                    g_arma_cieco = true;
                    g_pulsante_cieco = true;              // zittisce il rilevatore
                    g_candidato = false;                  // scarta un candidato gia' marcato
                }
            } else {
                if (alti < PULSANTE_STABILI) alti++;
                bassi = 0;
                if (alti >= PULSANTE_STABILI) gia_scattato = false;  // riarma
            }
        }

        if (blocks == 0) dsp_stage_set(DS_FIRST_BLOCK);
        watchdog_hw->scratch[5] = ++blocks;                      // blocchi elaborati
        if ((blocks % 31) == 0) gpio_xor_mask(1u << LED_PIN);    // battito a 1 Hz

        // Misuratore di livello per tarare il trimmer del microfono.
        // Obiettivo: dc vicino a 2048; parlando a mezzo metro escursione
        // 40-60%; battendo le mani vicino MAI min=0 o max=4095 (saturazione).
        if ((blocks % 62) == 0) {
            uint16_t mn = g_lvl_min, mx = g_lvl_max;
            say("LIVELLI min=%u max=%u dc=%.0f rms=%.1f escursione=%.0f%%%s\n",
                (unsigned)mn, (unsigned)mx, (double)g_lvl_dc, (double)g_lvl_rms,
                (double)(100.0f * (float)(mx - mn) / 4095.0f),
                (mn == 0 || mx == 4095) ? "  <<< SATURA, abbassa il guadagno" : "");
        }

        // Alimenta il watchdog solo se anche Task_ML e' vivo.
        TickType_t dl = g_ml_deadline;
        if (dl == 0 || (xTaskGetTickCount() - dl) < pdMS_TO_TICKS(ML_TIMEOUT_MS))
            watchdog_update();

        if (g_frames_filled >= FRAMES && ++since_last >= HOP_FRAMES) {
            since_last = 0;
            if (!g_ml_busy) {
                build_window();
                g_ml_busy = true;
                g_ml_deadline = xTaskGetTickCount();
                dsp_stage_set(DS_WINDOW_SENT);
                xSemaphoreGive(g_sem_window);
            } else {
                g_dropped++;
            }
        }
    }
}

// =============================================================================
//  TASK 2 - Inferenza (priorita' 2)
// =============================================================================
static const char* CATEGORIE[] = {
    "Attivita_Umana", "Ambiente_Urbano", "Veicoli", "Sirene_e_Urla",
    "Spari", "Incidente", "Vetri", "Fuochi"
};
#define NUM_CLASSES 8

// Stati della macchina deterministica, in ordine di GRAVITA' crescente:
// l'indice piu' alto vince quando piu' regole sono vere insieme.
enum Stato : uint8_t {
    ST_QUIETE = 0, ST_ATTIVITA_URBANA, ST_PERSONE, ST_FUOCHI_ART,
    ST_POSS_VANDALISMO, ST_EMERGENZA, ST_SPARO, ST_POSS_INCIDENTE
};
static const char* STATI[] = {
    "QUIETE", "ATTIVITA_URBANA", "PERSONE", "FUOCHI_ARTIFICIO",
    "POSSIBILE_VANDALISMO", "EMERGENZA", "SPARO", "POSSIBILE_INCIDENTE"
};
#define NUM_STATI 8

// Da questi stati in su si salva la clip di prova. FUOCHI_ARTIFICIO
// deliberatamente escluso: a Capodanno riempirebbe il canale.
static inline bool stato_pericoloso(int st) {
    return st == ST_SPARO || st == ST_EMERGENZA
        || st == ST_POSS_INCIDENTE || st == ST_POSS_VANDALISMO;
}

// Indici nelle CATEGORIE della CNN
#define C_UMANA   0
#define C_AMB     1
#define C_VEIC    2
#define C_SIRENE  3
#define C_SPARI   4
#define C_IMPATTO 5
#define C_VETRI   6
#define C_FUOCHI  7

// Nota: qui c'era un veto che forzava lo scenario quando la CNN rilevava uno
// scoppio ma il modello temporale diceva QUIETE. E' stato rimosso: sovrascrivere
// l'uscita della rete rendeva il grafico dello scenario difficile da leggere e
// da spiegare, perche' mescolava due decisioni prese con criteri diversi.
// Lo scenario mostrato ora e' esattamente quello del modello temporale, e le
// probabilita' istantanee restano disponibili nel campo "p" del JSON per
// qualunque logica di allerta si voglia costruire a valle.

// =============================================================================
//  MACCHINA A STATI DETERMINISTICA
//
//  Due strumenti, usati su classi diverse:
//    - livelli SMORZATI per le classi continue (ambiente, veicoli, voce,
//      sirene): conta la persistenza, e la media esponenziale toglie il
//      rumore di frame singolo
//    - EVENTI con refrattarieta' per le classi impulsive (spari, fuochi,
//      impatto, vetri): smorzare un impulso lo cancellerebbe
//
//  Le precondizioni sono la parte che toglie i falsi allarmi: senza veicoli
//  recenti non puo' esserci un incidente, e in silenzio nessun evento vale
//  perche' la normalizzazione al picco amplifica il rumore di fondo.
// =============================================================================
struct StatoMacchina {
    float    liv[NUM_CLASSES];          // livelli smorzati
    uint32_t ultimo_ev[NUM_CLASSES];    // ciclo dell'ultimo evento per classe
    uint8_t  n_ev[NUM_CLASSES];         // quanti eventi in totale
    uint32_t ev_recenti[NUM_CLASSES][MIN_FUOCHI + 2];  // ultimi istanti
    uint8_t  ev_w[NUM_CLASSES];
    uint32_t ultimo_veicoli;            // ultimo ciclo con veicoli presenti
    uint32_t ultimo_umana;              // ultimo ciclo con persone presenti
    uint32_t ultimo_vetri_forte;        // ultimo ciclo con Vetri sopra soglia autonoma
    uint32_t ultimo_cond_incidente;     // ultimo ciclo con contesto da incidente
    uint16_t sirene_consec;             // cicli consecutivi di sirene
    uint8_t  stato;
    uint32_t inizio_stato;
    uint32_t ciclo;
};

static void macchina_init(StatoMacchina* m) {
    memset(m, 0, sizeof(*m));
    m->stato = ST_QUIETE;
}

// Quanti eventi della classe negli ultimi 'finestra' cicli
static int eventi_in(const StatoMacchina* m, int c, uint32_t finestra) {
    int n = 0;
    for (int i = 0; i < MIN_FUOCHI + 2; i++) {
        uint32_t t = m->ev_recenti[c][i];
        if (t > 0 && (m->ciclo - t) <= finestra) n++;
    }
    return n;
}

// Soglia per classe: non tutte le classi meritano lo stesso sospetto.
static float soglia_di(int c) {
    if (c == C_IMPATTO || c == C_VETRI) return SOGLIA_DEBOLE;
    if (c == C_SIRENE)                  return SOGLIA_STRIDIO;
    return SOGLIA_EVENTO;
}

static void macchina_passo(StatoMacchina* m, const float* p, bool debole) {
    m->ciclo++;

    for (int i = 0; i < NUM_CLASSES; i++)
        m->liv[i] = ALPHA_LIV * p[i] + (1.0f - ALPHA_LIV) * m->liv[i];

    // Eventi con tempo morto, cosi' uno stesso suono conta una volta sola.
    // Sirene_e_Urla e' incluso perche' e' l'unica classe che riconosce lo
    // stridio di gomme, che qui vale come evento e non come livello continuo.
    const int impulsive[5] = { C_SPARI, C_FUOCHI, C_IMPATTO, C_VETRI, C_SIRENE };
    for (int k = 0; k < 5; k++) {
        int c = impulsive[k];
        // I Vetri sono l'unica classe accettata anche a finestra debole: un
        // vetro rotto in una stanza silenziosa e' proprio il caso che
        // interessa, e il filtro sul silenzio lo cancellerebbe.
        bool ammessa = (c == C_VETRI) ? true : !debole;
        if (ammessa && p[c] > soglia_di(c) &&
            (m->ciclo - m->ultimo_ev[c]) >= REFRATTARIO) {
            m->ultimo_ev[c] = m->ciclo;
            m->ev_recenti[c][m->ev_w[c]] = m->ciclo;
            m->ev_w[c] = (uint8_t)((m->ev_w[c] + 1) % (MIN_FUOCHI + 2));
            if (m->n_ev[c] < 255) m->n_ev[c]++;
        }
    }

    if (m->liv[C_VEIC]  > SOGLIA_LIVELLO) m->ultimo_veicoli = m->ciclo;
    if (m->liv[C_UMANA] > SOGLIA_LIVELLO) m->ultimo_umana   = m->ciclo;
    if (p[C_VETRI] > SOGLIA_VANDALISMO)  m->ultimo_vetri_forte = m->ciclo;

    // Sirene: serve CONTINUITA'. Un picco isolato e' quasi sempre la voce
    // scambiata male, quindi la soglia si alza quando domina la voce.
    float soglia_sirene = (m->liv[C_UMANA] > m->liv[C_SIRENE])
                        ? SOGLIA_LIVELLO * 1.8f : SOGLIA_LIVELLO;
    if (!debole && m->liv[C_SIRENE] > soglia_sirene) m->sirene_consec++;
    else m->sirene_consec = 0;

    // ---- stato di fondo ----
    int cand = ST_QUIETE;
    if (!debole) {
        if (m->liv[C_UMANA] >= m->liv[C_VEIC] && m->liv[C_UMANA] >= m->liv[C_AMB]
            && m->liv[C_UMANA] > SOGLIA_LIVELLO)
            cand = ST_PERSONE;
        else if (m->liv[C_VEIC] > SOGLIA_LIVELLO || m->liv[C_AMB] > SOGLIA_LIVELLO)
            cand = ST_ATTIVITA_URBANA;
    }

    // ---- regole sugli eventi, dalla meno alla piu' grave ----
    int scoppi = eventi_in(m, C_SPARI, FIN_FUOCHI) + eventi_in(m, C_FUOCHI, FIN_FUOCHI);
    if (scoppi >= MIN_FUOCHI && cand < ST_FUOCHI_ART) cand = ST_FUOCHI_ART;

    if (m->sirene_consec >= SIRENE_CONTINUE && cand < ST_EMERGENZA)
        cand = ST_EMERGENZA;

    // Sparo: uno e uno solo scoppio nella finestra di isolamento
    int scoppi_iso = eventi_in(m, C_SPARI, ISOLAMENTO) + eventi_in(m, C_FUOCHI, ISOLAMENTO);
    if (scoppi_iso == 1 && scoppi < MIN_FUOCHI && cand < ST_SPARO) cand = ST_SPARO;

    // ---- incidente ----
    // Un impatto in una scena deserta e' un rumore qualsiasi: serve che
    // qualcosa fosse presente un attimo prima, veicoli o persone.
    bool contesto =
        ((m->ultimo_veicoli > 0) && ((m->ciclo - m->ultimo_veicoli) <= MEM_CONTESTO)) ||
        ((m->ultimo_umana   > 0) && ((m->ciclo - m->ultimo_umana)   <= MEM_CONTESTO));

    // Lo stridio di gomme, riconosciuto come Sirene_e_Urla.
    int stridio = eventi_in(m, C_SIRENE, FIN_INCIDENTE);

    // Indizi di impatto: si contano le CLASSI DIVERSE che si sono attivate,
    // non gli eventi. Un impatto forte puo' presentarsi come Incidente,
    // Vetri, Spari o Fuochi a seconda di come la CNN lo interpreta.
    const int imp_cls[4] = { C_IMPATTO, C_VETRI, C_SPARI, C_FUOCHI };
    int n_imp = 0;
    for (int k = 0; k < 4; k++)
        if (eventi_in(m, imp_cls[k], FIN_INCIDENTE) > 0) n_imp++;

    bool cond_incidente = contesto && stridio > 0 && n_imp >= 1;
    if (cond_incidente) {
        m->ultimo_cond_incidente = m->ciclo;
        if (cand < ST_POSS_INCIDENTE) cand = ST_POSS_INCIDENTE;
    }

    // ---- vandalismo ----
    // Vetri netti FUORI da un contesto di incidente. L'esclusione e' esplicita
    // e non affidata alla sola priorita': i vetri di un tamponamento non sono
    // vandalismo, e vanno segnalati per quello che sono.
    bool vetri_forti = (m->ultimo_vetri_forte > 0) &&
                       ((m->ciclo - m->ultimo_vetri_forte) <= FIN_INCIDENTE);
    // L'esclusione vale anche per un po' DOPO: lo stridio esce dalla finestra
    // prima dei vetri, e senza memoria del contesto un incidente scivolerebbe
    // in vandalismo proprio mentre i vetri sono ancora recenti.
    bool incidente_recente = (m->ultimo_cond_incidente > 0) &&
        ((m->ciclo - m->ultimo_cond_incidente) <= FIN_INCIDENTE * 2);
    if (!incidente_recente && vetri_forti && cand < ST_POSS_VANDALISMO)
        cand = ST_POSS_VANDALISMO;

    // ---- isteresi: uno stato puo' essere promosso subito, abbassato solo
    //      dopo DURATA_MIN cicli ----
    if (cand >= m->stato || (m->ciclo - m->inizio_stato) >= DURATA_MIN) {
        if (cand != m->stato) {
            m->stato = (uint8_t)cand;
            m->inizio_stato = m->ciclo;
        }
    }
}

static void Task_ML(void* pv) {
    (void)pv;

    stage_set(ST_ML_START);
    watchdog_hw->scratch[6] = 0;

    stage_set(ST_TFLM_GETMODEL);
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        say("ERRORE: schema %d, atteso %d\n", (int)model->version(), TFLITE_SCHEMA_VERSION);
        vTaskDelete(NULL);
    }

    // Operatori della rete senza depthwise: CONV_2D, MAX_POOL_2D,
    // AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX.
    // Gli altri restano registrati come rete di sicurezza.
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();
    resolver.AddAdd();

    alignas(16) static uint8_t tensor_arena[ARENA_SIZE];
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA_SIZE);

    stage_set(ST_TFLM_ALLOCATE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        say("ERRORE FATALE: AllocateTensors fallito. Aumenta ARENA_SIZE oppure\n"
            "manca un operatore nel resolver.\n");
        vTaskDelete(NULL);
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    say("Modello: %u byte, arena usata %u/%u\n",
        g_model_len, (unsigned)interpreter.arena_used_bytes(), (unsigned)ARENA_SIZE);
    say("Input: %d assi, %u byte, scale=%.8f zp=%d\n",
        input->dims->size, (unsigned)input->bytes,
        (double)input->params.scale, (int)input->params.zero_point);

    if (input->bytes != (size_t)(N_MELS * FRAMES)) {
        say("ERRORE: il modello vuole %u byte di input, il firmware ne produce %d.\n"
            "Allinea FRAMES (ora %d) a extract_features.py.\n",
            (unsigned)input->bytes, N_MELS * FRAMES, FRAMES);
        vTaskDelete(NULL);
    }

    static StatoMacchina mac;
    macchina_init(&mac);
    uint32_t ultimo_dump = 0;
    uint32_t candidato_da = 0;


    // Inferenza a vuoto. Se si blocca qui, il watchdog riavvia e lo segnala.
    memset(input->data.int8, input->params.zero_point, input->bytes);
    g_ml_deadline = xTaskGetTickCount();
    stage_set(ST_WARMUP_ENTER);
    uint32_t tw = time_us_32();
    TfLiteStatus warm = interpreter.Invoke();
    tw = time_us_32() - tw;
    stage_set(ST_WARMUP_DONE);
    g_ml_deadline = 0;
    say("Invoke a vuoto completata in %u us (stato=%d)\n", (unsigned)tw, (int)warm);

    const float in_scale  = input->params.scale;
    const int   in_zp     = input->params.zero_point;
    const float out_scale = output->params.scale;
    const int   out_zp    = output->params.zero_point;

    for (;;) {
        stage_set(ST_WAIT_WINDOW);
        xSemaphoreTake(g_sem_window, portMAX_DELAY);

        int8_t* dst = input->data.int8;
        for (int i = 0; i < N_MELS * FRAMES; i++) {
            float v = g_window_db[i];
            if (v < MIN_DB) v = MIN_DB;
            if (v > MAX_DB) v = MAX_DB;
            v = (v - MIN_DB) / (MAX_DB - MIN_DB);
            int q = (int)lrintf(v / in_scale) + in_zp;
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            dst[i] = (int8_t)q;
        }

        stage_set(ST_INVOKE_ENTER);
        uint32_t t0 = time_us_32();
        TfLiteStatus st = interpreter.Invoke();
        uint32_t dt = time_us_32() - t0;
        g_ml_deadline = 0;
        stage_set(ST_RUNNING);
        watchdog_hw->scratch[6]++;

        if (st != kTfLiteOk) {
            say("ERRORE: Invoke fallito (%d)\n", (int)st);
            g_ml_busy = false;
            continue;
        }

        int8_t* q = output->data.int8;
        int c0 = 0, c1 = 0, c2 = 0, p0 = -129, p1 = -129, p2 = -129;
        for (int i = 0; i < NUM_CLASSES; i++) {
            int v = q[i];
            if (v > p0)      { p2=p1; c2=c1; p1=p0; c1=c0; p0=v; c0=i; }
            else if (v > p1) { p2=p1; c2=c1; p1=v;  c1=i; }
            else if (v > p2) { p2=v;  c2=i; }
        }
        g_ml_busy = false;

        float prob0 = ((float)p0 - out_zp) * out_scale;

        // ---- macchina a stati deterministica ----
        float probs[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++)
            probs[i] = ((float)q[i] - out_zp) * out_scale;

        // Finestra cieca dopo il clic del pulsante. Si riusa il percorso della
        // "finestra debole": la macchina a stati scarta gia' gli eventi
        // impulsivi in quel caso, quindi non serve logica nuova ne' toccare
        // nessuna soglia. La CNN continua a classificare e il JSON esce lo
        // stesso, ma marcato: si vede cosa il Pico ha sentito senza che quel
        // rumore si trasformi in uno stato.
        static uint32_t cieco = 0;
        if (g_arma_cieco) { cieco = PULSANTE_CIECO; g_arma_cieco = false; }
        bool ignora = (cieco > 0);
        if (cieco) cieco--;
        if (cieco == 0) g_pulsante_cieco = false;   // riabilita il rilevatore rapido

        macchina_passo(&mac, probs, g_finestra_debole || ignora);

        // ---- innesco della clip di prova ----
        // Due inneschi indipendenti, ed e' necessario che siano due.
        //
        // Il rilevatore rapido marca un candidato entro 32 ms dal picco e
        // serve a non perdere l'attacco. Ma per un incidente lo stridio e' il
        // primo suono forte: fa scattare il candidato quando lo stato non e'
        // ancora pericoloso, e il candidato scade prima che l'urto arrivi.
        // Peggio, lo stridio alza la media mobile dell'energia e l'urto
        // successivo puo' non superare piu' la soglia del rilevatore.
        //
        // Per questo scatta anche il PASSAGGIO a uno stato pericoloso. Il
        // buffer conserva 4 s e ne riversiamo 2: anche innescando al cambio
        // di stato, l'evento che l'ha causato e' ancora dentro.
        static uint8_t stato_prec = ST_QUIETE;
        bool salita = stato_pericoloso(mac.stato) && !stato_pericoloso(stato_prec);
        stato_prec = mac.stato;

        if (g_candidato) {
            if (candidato_da == 0) candidato_da = mac.ciclo;
            else if ((mac.ciclo - candidato_da) > CAND_VALIDO) {
                g_candidato = false;
                candidato_da = 0;
            }
        }
        if (salita || (g_candidato && stato_pericoloso(mac.stato))) {
            bool libero = !g_clip_congelato && !g_clip_pronta;
            bool passato = (ultimo_dump == 0) ||
                           ((mac.ciclo - ultimo_dump) >= TEMPO_MORTO);
            if (libero && passato) {
                // Congela: da qui Task_DSP smette di scrivere nel buffer,
                // quindi il WAV non risultera' un miscuglio di due momenti.
                g_clip_congelato = true;
                g_clip_pronta = true;
                ultimo_dump = mac.ciclo;
            } else {
                // Sovraccarico: si SCARTA il lavoro in eccesso invece di
                // accodarlo. Accodare significherebbe riversare, fra un
                // minuto, eventi vecchi di un minuto.
                g_clip_perse++;
            }
            g_candidato = false;
            candidato_da = 0;
        }

        char sc[64];
        snprintf(sc, sizeof(sc), ",\"stato\":\"%s\",\"clip_perse\":%u,\"cieco\":%d",
                 STATI[mac.stato], (unsigned)g_clip_perse, ignora ? 1 : 0);

        if (g_finestra_debole) {
            say("{\"t1\":\"Silenzio\",\"p1\":1.000%s,\"dsp_us\":%u,"
                "\"dsp_max_us\":%u,\"inf_us\":%u,\"adc_ovr\":%u,"
                "\"drop\":%u}\n", sc,
                (unsigned)g_dsp_us_last, (unsigned)g_dsp_us_max, (unsigned)dt,
                (unsigned)g_adc_overruns, (unsigned)g_dropped);
            continue;
        }

        // Vettore completo delle otto probabilita': e' l'ingresso che servira'
        // a uno strato temporale (media mobile, macchina a stati o rete
        // ricorrente) per ragionare sulla sequenza invece che sul singolo
        // istante. Le prime tre restano per leggibilita' umana.
        char pv[96];
        int  n = 0;
        for (int i = 0; i < NUM_CLASSES && n < (int)sizeof(pv) - 8; i++)
            n += snprintf(pv + n, sizeof(pv) - n, "%s%.3f",
                          i ? "," : "",
                          (double)(((float)q[i] - out_zp) * out_scale));

        say("{\"t1\":\"%s\",\"p1\":%.3f,\"t2\":\"%s\",\"p2\":%.3f,"
            "\"t3\":\"%s\",\"p3\":%.3f,\"incerto\":%d,\"p\":[%s]%s,"
            "\"dsp_us\":%u,\"dsp_max_us\":%u,\"inf_us\":%u,"
            "\"adc_ovr\":%u,\"drop\":%u}\n",
            CATEGORIE[c0], (double)prob0,
            CATEGORIE[c1], (double)(((float)p1 - out_zp) * out_scale),
            CATEGORIE[c2], (double)(((float)p2 - out_zp) * out_scale),
            (prob0 < SOGLIA_CONFIDENZA) ? 1 : 0, pv, sc,
            (unsigned)g_dsp_us_last, (unsigned)g_dsp_us_max, (unsigned)dt,
            (unsigned)g_adc_overruns, (unsigned)g_dropped);
    }
}

// =============================================================================
//  TASK 3 - Uscita seriale (priorita' 1)
// =============================================================================
static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Riversa gli ultimi n campioni del buffer circolare, in ordine cronologico.
// Una sola funzione per entrambi gli usi: 3,5 s per la clip di allerta,
// 1 s per il comando 'r' della raccolta dataset.
// Gira in Task_Serial a priorita' 1, quindi puo' durare secondi senza mai
// ritardare DSP o inferenza: un task a priorita' inferiore non compare
// nell'analisi dei tempi di risposta di quelli sopra.
static void dump_finestra(uint32_t n, const char* tipo) {
    printf("#AUDIO START sr=%d n=%u bits=8 tipo=%s\n",
           SAMPLE_RATE, (unsigned)n, tipo);
    char     riga[65];
    int      c = 0, nt = 0;
    uint8_t  tre[3];
    uint32_t inizio = (g_clip_w + CLIP_SAMPLES - n) % CLIP_SAMPLES;

    for (uint32_t k = 0; k < n; k++) {
        tre[nt++] = (uint8_t)(int8_t)g_clip[(inizio + k) % CLIP_SAMPLES];
        if (nt < 3 && k + 1 < n) continue;
        uint32_t v = (uint32_t)tre[0] << 16;
        if (nt > 1) v |= (uint32_t)tre[1] << 8;
        if (nt > 2) v |= (uint32_t)tre[2];
        riga[c++] = B64[(v >> 18) & 63];
        riga[c++] = B64[(v >> 12) & 63];
        riga[c++] = (nt > 1) ? B64[(v >> 6) & 63] : '=';
        riga[c++] = (nt > 2) ? B64[v & 63] : '=';
        nt = 0;
        if (c >= 64) { riga[c] = 0; puts(riga); c = 0; }
    }
    if (c) { riga[c] = 0; puts(riga); }
    printf("#AUDIO END\n");
    fflush(stdout);
}

static void Task_Serial(void* pv) {
    (void)pv;
    uint8_t idx;
    for (;;) {
        // Comandi dall'host. 'r' riversa l'ultimo secondo registrato.
        int ch = getchar_timeout_us(0);
        if ((ch == 'r' || ch == 'R') && !g_clip_pronta) g_rec_richiesta = true;

        if (g_pin_cambiato >= 0) {
            printf("#PIN gpio%d=%d\n", PULSANTE_GPIO, g_pin_cambiato);
            fflush(stdout);
            g_pin_cambiato = -1;
        }

        if (g_demo_richiesta) {
            g_demo_richiesta = false;
            printf("#DEMO n=%u\n", (unsigned)g_demo_n);
            fflush(stdout);
        }

        if (g_rec_richiesta) {
            g_clip_congelato = true;      // ferma la scrittura durante il dump
            dump_finestra(REC_CAMPIONI, "rec");
            g_clip_congelato = false;
            g_rec_richiesta = false;
        }

        if (g_clip_pronta) {
            dump_finestra(CLIP_PRE, "clip");
            g_clip_pronta = false;
            g_clip_congelato = false;     // riprende la scrittura circolare
        }

        // Timeout breve: la coda va svuotata ma bisogna anche leggere i comandi
        if (xQueueReceive(g_log_q, &idx, pdMS_TO_TICKS(20)) == pdTRUE) {
            // Unico punto del programma in cui si scrive su stdout dai task.
            fputs(g_log_buf[idx], stdout);
            fflush(stdout);
        }
    }
}

// =============================================================================
//  Hook diagnostici. Registrano la causa e lasciano riavviare il watchdog,
//  cosi' il messaggio esce comunque al riavvio successivo.
// =============================================================================
extern "C" void vApplicationStackOverflowHook(TaskHandle_t t, char* name) {
    (void)t; (void)name;
    watchdog_hw->scratch[1] = 0xDEAD0001u;
    for (;;) tight_loop_contents();
}

extern "C" void vApplicationMallocFailedHook(void) {
    watchdog_hw->scratch[1] = 0xDEAD0002u;
    for (;;) tight_loop_contents();
}

extern "C" void __attribute__((naked)) isr_hardfault(void) {
    __asm volatile(
        "movs r0, #4            \n"
        "mov  r1, lr            \n"
        "tst  r0, r1            \n"
        "beq  1f                \n"
        "mrs  r0, psp           \n"
        "b    hardfault_report  \n"
        "1:                     \n"
        "mrs  r0, msp           \n"
        "b    hardfault_report  \n"
    );
}

extern "C" void hardfault_report(uint32_t* sp) {
    watchdog_hw->scratch[1] = 0xDEAD0003u;
    watchdog_hw->scratch[2] = sp[6];         // PC al momento del fault
    watchdog_hw->scratch[3] = sp[5];         // LR
    for (;;) tight_loop_contents();
}

// =============================================================================
int main() {
    // 200 MHz: il blocco di Invoke() era DepthwiseConv2D, non l'overclock, quindi
    // il vincolo dei 133 MHz non serve piu'. Con hop 256 i frame raddoppiano e
    // il DSP passa da ~9 a ~19 ms sui 32 disponibili per blocco.
    set_sys_clock_khz(200000, true);

    // GPIO23 e' il pin PS del regolatore switching: alto = modalita' PWM
    // forzata. In PFM (default a basso carico) il ripple sul rail 3V3 finisce
    // nel microfono e nel riferimento dell'ADC. Questa e' la singola riga con
    // il miglior rapporto risultato/sforzo per la qualita' delle letture.
    gpio_init(23);
    gpio_set_dir(23, GPIO_OUT);
    gpio_put(23, 1);

    stdio_init_all();
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    gpio_init(PULSANTE_GPIO);
    gpio_set_dir(PULSANTE_GPIO, GPIO_IN);
    gpio_pull_up(PULSANTE_GPIO);      // pulsante verso GND: a riposo legge 1

    bool     from_wdt   = watchdog_caused_reboot();
    uint32_t last_stage = stage_get();
    uint32_t last_dsp   = dsp_stage_get();
    uint32_t n_blocks   = watchdog_hw->scratch[5];
    uint32_t n_infer    = watchdog_hw->scratch[6];
    uint32_t fault_code = watchdog_hw->scratch[1];
    uint32_t fault_pc   = watchdog_hw->scratch[2];
    uint32_t fault_lr   = watchdog_hw->scratch[3];
    watchdog_hw->scratch[1] = 0;

    stage_set(ST_BOOT);
    dsp_stage_set(0);
    watchdog_hw->scratch[5] = 0;
    watchdog_hw->scratch[6] = 0;

    // Attesa dell'host, fino a 20 s. Se il sistema si riavvia in ciclo, questa
    // finestra garantisce che prima o poi read_pico.py sia gia' in ascolto.
    for (int i = 0; i < 200 && !stdio_usb_connected(); i++) {
        sleep_ms(100);
        if ((i % 5) == 0) gpio_xor_mask(1u << LED_PIN);
    }
    gpio_put(LED_PIN, 0);
    sleep_ms(200);
    stage_set(ST_STDIO);

    printf("\n========================================\n");
    printf("Smart City Audio - clock %u kHz - TensorFlow Lite Micro\n",
           clock_get_hz(clk_sys) / 1000);
    if (from_wdt) {
        printf(">>> RIAVVIO CAUSATO DAL WATCHDOG <<<\n");
        printf(">>> Task_ML  fase %u: %s\n", last_stage, stage_name(last_stage));
        printf(">>> Task_DSP fase %u: %s\n", last_dsp,   dsp_stage_name(last_dsp));
        printf(">>> Blocchi audio elaborati: %u   Inferenze completate: %u\n",
               n_blocks, n_infer);
        switch (fault_code) {
            case 0xDEAD0001u:
                printf(">>> Causa: STACK OVERFLOW\n"); break;
            case 0xDEAD0002u:
                printf(">>> Causa: heap FreeRTOS esaurito\n"); break;
            case 0xDEAD0003u:
                printf(">>> Causa: HARD FAULT  PC=0x%08lx LR=0x%08lx\n",
                       (unsigned long)fault_pc, (unsigned long)fault_lr); break;
            default:
                printf(">>> Causa: blocco senza eccezione (deadlock o loop infinito)\n"); break;
        }
    } else {
        printf("Avvio normale.\n");
    }
    printf("========================================\n");
    fflush(stdout);

    g_sem_block  = xSemaphoreCreateBinary();
    g_sem_window = xSemaphoreCreateBinary();
    QueueHandle_t log_q = xQueueCreate(LOG_LINES, sizeof(uint8_t));
    if (!g_sem_block || !g_sem_window || !log_q) {
        printf("ERRORE: heap FreeRTOS insufficiente per gli oggetti di sincronizzazione\n");
        fflush(stdout);
        while (true) tight_loop_contents();
    }

    // Stack in PAROLE (x4 byte): 3072=12 KB, 8192=32 KB, 1024=4 KB
    BaseType_t ok = pdPASS;
    ok &= xTaskCreate(Task_DSP,    "DSP", 3072, NULL, 3, NULL);
    ok &= xTaskCreate(Task_ML,     "ML",  8192, NULL, 2, NULL);
    ok &= xTaskCreate(Task_Serial, "SER", 1024, NULL, 1, NULL);
    if (ok != pdPASS) {
        printf("ERRORE: xTaskCreate fallita, configTOTAL_HEAP_SIZE insufficiente\n");
        fflush(stdout);
        while (true) tight_loop_contents();
    }

    printf("Pulsante su GPIO%d: a riposo legge %d (deve essere 1; se legge 0 "
           "il contatto e' chiuso, cioe' le due zampe collegate sono quelle "
           "gia' unite internamente)\n",
           PULSANTE_GPIO, gpio_get(PULSANTE_GPIO));
    printf("Avvio scheduler, watchdog a %d ms.\n\n", WATCHDOG_MS);
    fflush(stdout);
    g_log_q = log_q;      // da qui in poi say() accoda invece di stampare
    watchdog_enable(WATCHDOG_MS, 1);

    vTaskStartScheduler();
    while (true) tight_loop_contents();
}
