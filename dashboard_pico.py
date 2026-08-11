"""
Cruscotto in tempo reale per il classificatore audio sul Raspberry Pi Pico.

Mostra le due logiche del sistema:

  1. CNN  - mappa di calore con TUTTE e otto le probabilita' ogni 0,5 s.
            Mostrare solo la classe vincente nascondeva proprio cio' su cui
            la macchina a stati decide: una classe al 60% che perde contro
            un'altra al 65% non compariva, eppure aveva superato la soglia
            di evento e armato la macchina.
  2. macchina a stati  - scenario da regole deterministiche, con livelli
                         smorzati per le classi continue, eventi con
                         refrattarieta' per quelle impulsive, e precondizioni
                         (senza veicoli recenti non puo' esserci un incidente)

Il modello temporale appreso e' stato rimosso dal firmware: si fermava al 76%
sugli scenari, con f1 0,54 sullo sparo singolo, e la macchina a stati copre
meglio proprio i casi puntuali dove quello sbagliava.

Salva anche le clip di prova che il Pico riversa quando riconosce un evento
pericoloso: intercetta i marcatori #AUDIO START / #AUDIO END e scrive un WAV.

Requisiti:
    pip install matplotlib pyserial

Uso:
    python dashboard_pico.py
    python dashboard_pico.py --finestra 120 --aggrega 5
    python dashboard_pico.py --console
"""
import argparse
import base64
import json
import os
import queue
import struct
import threading
import time
import wave
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports

BAUD_RATE = 115200
PICO_VID = "2E8A"
CARTELLA_CLIP = "clip_allerta"
CARTELLA_DEMO = "samples"       # campioni da riprodurre col pulsante
# Deve coincidere con PULSANTE_CIECO nel firmware: 3 cicli da 0,5 s.
DURATA_CIECA = 1.5
# Il campione parte DOPO la finestra cieca, altrimenti il suo attacco - la
# parte piu' informativa - cadrebbe proprio nei cicli che il firmware scarta.
# Legato a DURATA_CIECA e non fissato a mano: se un giorno cambi la finestra,
# il ritardo la segue senza doversene ricordare.
MARGINE_DEMO = 0.3

# Ordine con cui il firmware manda le probabilita' nel campo "p".
# Deve coincidere con l'array CATEGORIE del main.cpp.
CLASSI_CNN = [
    "Attivita_Umana", "Ambiente_Urbano", "Veicoli", "Sirene_e_Urla",
    "Spari", "Incidente", "Vetri", "Fuochi",
]
# Ordine di RILEVANZA crescente sull'asse: dal basso in alto si peggiora.
RILEVANZA = [
    "Ambiente_Urbano", "Veicoli", "Attivita_Umana",
    "Vetri", "Fuochi", "Sirene_e_Urla", "Incidente", "Spari",
]
# permutazione dall'ordine del firmware a quello di rilevanza
PERM = [CLASSI_CNN.index(c) for c in RILEVANZA]

# Nome mostrato a schermo: la classe "Sirene_e_Urla" riconosce bene anche lo
# stridio di gomme, ed e' su quello che la macchina a stati la usa per gli
# incidenti. Il nome tecnico resta quello del firmware, cambia l'etichetta.
NOMI = {"Sirene_e_Urla": "Sirene_Urla_Gomme"}
def etichetta(c):
    return NOMI.get(c, c)

# Grafico della sola classe vincente: qui serve anche il livello "Silenzio",
# che nella mappa di calore e' semplicemente una colonna nera.
LINEA = ["Silenzio"] + RILEVANZA

# Sopra questa probabilita' il firmware considera "successo" un evento
# impulsivo: la riga di contorno sulla mappa segna proprio quel livello.
SOGLIA_EVENTO = 0.50
PASSO_GRIGLIA = 0.5      # una colonna ogni inferenza

# Ordine di PERICOLOSITA' crescente: il primo elemento sta IN FONDO al
# grafico, l'ultimo in cima. Questo e' solo l'ordine di visualizzazione;
# la priorita' con cui gli stati si contendono la decisione e' definita
# dall'enum nel firmware, ed e' una cosa distinta.
STATI = [
    "QUIETE", "ATTIVITA_URBANA", "PERSONE", "FUOCHI_ARTIFICIO",
    "POSSIBILE_VANDALISMO", "EMERGENZA", "POSSIBILE_INCIDENTE", "SPARO",
]

ALLERTA_CAT = RILEVANZA.index("Vetri")
ALLERTA_LINEA = LINEA.index("Vetri")
ALLERTA_STATO = STATI.index("EMERGENZA")


class Riproduttore(threading.Thread):
    """Riproduce i campioni della cartella demo quando arriva il marcatore
    #DEMO dal pulsante sulla breadboard.

    Gira in un thread proprio: la riproduzione dura secondi e sul thread
    principale bloccherebbe l'animazione dei grafici. I file vengono percorsi
    in ordine alfabetico e non a caso, cosi' durante una presentazione si sa
    sempre quale sara' il prossimo.
    """

    def __init__(self, cartella, ritardo):
        super().__init__(daemon=True)
        self.cartella = cartella
        self.ritardo = ritardo
        self.coda = queue.Queue(maxsize=1)
        self.indice = 0
        self.corrente = None
        self.in_corso = False
        self.disponibile = True
        try:
            import sounddevice as sd
            import soundfile as sf
            self.sd, self.sf = sd, sf
        except Exception as e:
            # Non solo ImportError: se manca la libreria di sistema PortAudio,
            # sounddevice solleva OSError all'import. In ogni caso il cruscotto
            # deve continuare a funzionare, solo senza riproduzione.
            self.disponibile = False
            print(f"[demo] riproduzione non disponibile ({e}).")
            print("       pip install sounddevice soundfile")
            print("       su Linux serve anche: sudo apt install libportaudio2")

    def elenco(self):
        if not os.path.isdir(self.cartella):
            return []
        return sorted(f for f in os.listdir(self.cartella)
                      if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3")))

    def prossimo_nome(self):
        """Nome del file che verra' riprodotto, senza avanzare l'indice."""
        files = self.elenco()
        return files[self.indice % len(files)] if files else "?"

    def richiedi(self):
        """Ignora la richiesta se una riproduzione e' gia' in corso: accodarle
        significherebbe sentire, fra dieci secondi, un campione chiesto adesso."""
        if self.in_corso:
            print("[demo] gia' in riproduzione, richiesta ignorata")
            return
        try:
            self.coda.put_nowait(True)
        except queue.Full:
            pass

    def run(self):
        while True:
            self.coda.get()
            if not self.disponibile:
                print("[demo] riproduzione non disponibile")
                continue
            files = self.elenco()
            if not files:
                print(f"[demo] nessun file in '{self.cartella}/' "
                      f"(percorso assoluto: {os.path.abspath(self.cartella)})")
                continue
            nome = files[self.indice % len(files)]
            self.indice += 1
            self.in_corso = True
            self.corrente = nome
            if self.ritardo > 0:
                time.sleep(self.ritardo)
            try:
                dati, sr = self.sf.read(os.path.join(self.cartella, nome),
                                        dtype="float32", always_2d=True)
                print(f"[demo] riproduco {nome}  {dati.shape[0]/sr:.1f}s "
                      f"{sr} Hz  ({self.indice}/{len(files)}) "
                      f"dopo {self.ritardo:.1f}s di attesa")
                # Stream esplicito invece di sd.play(): l'apertura e chiusura
                # sono deterministiche e la scrittura blocca fino alla fine.
                # sd.play() lascia la gestione dello stream alla libreria, e su
                # Windows chiamate ravvicinate producono riproduzioni saltuarie.
                with self.sd.OutputStream(samplerate=sr,
                                          channels=dati.shape[1],
                                          dtype="float32") as st:
                    st.write(dati)
                print(f"[demo] fine {nome}")
            except Exception as e:
                print(f"[demo] errore su {nome}: {type(e).__name__}: {e}")
            finally:
                self.in_corso = False
                self.corrente = None


class LettoreSeriale(threading.Thread):
    """Legge la porta in continuazione. Il thread dedicato serve anche a non
    far mai riempire il buffer USB: se nessuno legge, il firmware si blocca
    in scrittura e il riversamento delle clip non parte."""

    def __init__(self):
        super().__init__(daemon=True)
        self.righe = queue.Queue()
        self.stato = "in attesa del Pico"

    @staticmethod
    def trova_porta():
        for p in serial.tools.list_ports.comports():
            if PICO_VID in p.hwid or "ttyACM" in p.device:
                return p.device
        return None

    def run(self):
        while True:
            porta = self.trova_porta()
            if porta is None:
                self.stato = "Pico non collegato"
                time.sleep(0.5)
                continue
            try:
                with serial.Serial(porta, BAUD_RATE, timeout=1.0) as sp:
                    self.stato = f"connesso a {porta}"
                    while True:
                        raw = sp.readline()
                        if not raw:
                            continue
                        linea = raw.decode("utf-8", errors="replace").strip()
                        if linea:
                            self.righe.put(linea)
            except (serial.SerialException, OSError, PermissionError):
                self.stato = "riconnessione..."
                time.sleep(1.0)


class RaccoglitoreClip:
    """Intercetta i marcatori del riversamento e scrive il WAV.

    Il Pico manda i campioni a 8 bit (gamma dinamica ridotta per stare nei
    32 KB gia' allocati): qui si riportano a 16 bit per il file."""

    def __init__(self, cartella):
        self.cartella = cartella
        os.makedirs(cartella, exist_ok=True)
        self.dentro = False
        self.righe = []
        self.sr = 16000
        self.bits = 16
        self.salvate = 0
        self.ultima = None

    def linea(self, testo):
        """True se la riga fa parte di un riversamento (e va nascosta)."""
        if testo.startswith("#AUDIO START"):
            self.dentro, self.righe = True, []
            for tok in testo.split():
                if tok.startswith("sr="):
                    self.sr = int(tok[3:])
                elif tok.startswith("bits="):
                    self.bits = int(tok[5:])
            return True
        if testo.startswith("#AUDIO END"):
            self.dentro = False
            self._salva()
            return True
        if self.dentro:
            self.righe.append(testo)
            return True
        return False

    def _salva(self):
        try:
            grezzi = base64.b64decode("".join(self.righe))
        except Exception:
            return
        if self.bits == 8:
            camp = [c * 256 for c in struct.unpack(f"<{len(grezzi)}b", grezzi)]
        else:
            n = len(grezzi) // 2
            camp = [c * 16 for c in struct.unpack(f"<{n}h", grezzi[:n * 2])]
        camp = [max(-32768, min(32767, c)) for c in camp]

        nome = time.strftime("clip_%Y%m%d_%H%M%S.wav")
        percorso = os.path.join(self.cartella, nome)
        with wave.open(percorso, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(struct.pack(f"<{len(camp)}h", *camp))
        self.salvate += 1
        self.ultima = nome
        print(f"[clip] salvata {percorso}  ({len(camp)/self.sr:.1f} s)")


class Cruscotto:
    def __init__(self, finestra, aggrega, aggrega_classe, console, ritardo):
        self.finestra, self.aggrega, self.console = finestra, aggrega, console
        # La classe vincente si aggrega piu' fitto degli stati: cambia a ogni
        # inferenza, mentre uno stato dura secondi per costruzione.
        self.aggrega_classe = aggrega_classe
        self.t0 = time.time()
        self.prob, self.vinc, self.stato = deque(), deque(), deque()
        self.demo_eventi = deque()   # (t_inizio, t_fine, nome_file)
        self.bande = []              # rettangoli disegnati, da rimuovere ogni giro
        self.testi = []
        self.ultimo = {}

        self.lettore = LettoreSeriale()
        self.lettore.start()
        self.clip = RaccoglitoreClip(CARTELLA_CLIP)
        self.demo = Riproduttore(CARTELLA_DEMO, ritardo)
        self.demo.start()

        plt.style.use("dark_background")
        self.fig, (self.ax1, self.axL, self.ax2) = plt.subplots(
            3, 1, figsize=(13, 10), sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1, 1]})
        self.fig.canvas.manager.set_window_title("Smart City Audio - Pico")

        # --- mappa di calore delle otto probabilita' ---
        self.n_col = max(8, int(finestra / PASSO_GRIGLIA))
        self.griglia = np.zeros((len(RILEVANZA), self.n_col), dtype=float)
        self.img = self.ax1.imshow(
            self.griglia, aspect="auto", origin="lower", interpolation="nearest",
            cmap="magma", vmin=0.0, vmax=1.0,
            extent=[0, finestra, -0.5, len(RILEVANZA) - 0.5])
        self.ax1.set_yticks(range(len(RILEVANZA)))
        self.ax1.set_yticklabels([etichetta(c) for c in RILEVANZA], fontsize=8)
        self.ax1.set_title("1. CNN — probabilita' di tutte le classi (0,5 s)",
                           fontsize=10, loc="left", color="#e8eaed")
        self.ax1.axhline(ALLERTA_CAT - 0.5, color="#8ab4f8", alpha=0.5, lw=0.8)
        cb = self.fig.colorbar(self.img, ax=self.ax1, pad=0.01, fraction=0.025)
        cb.set_label("probabilita'", fontsize=8, color="#9aa0a6")
        cb.ax.tick_params(labelsize=7, colors="#9aa0a6")

        self._asse(self.axL, [etichetta(c) for c in LINEA],
                   f"2. CNN — classe vincente (max su {aggrega_classe:g} s)",
                   ALLERTA_LINEA)
        self.p_fine, = self.axL.plot([], [], "o", ms=3, alpha=0.30, color="#8ab4f8")
        self.l_vinc, = self.axL.plot([], [], drawstyle="steps-post",
                                     lw=2.2, color="#8ab4f8")

        self._asse(self.ax2, STATI,
                   f"3. Macchina a stati — regole deterministiche "
                   f"(max su {aggrega:g} s)", ALLERTA_STATO)
        self.ax2.set_xlabel("secondi")
        self.l_stato, = self.ax2.plot([], [], drawstyle="steps-post",
                                      lw=2.2, color="#81c995")

        self.testo = self.fig.text(0.01, 0.965, "", fontsize=9,
                                   color="#9aa0a6", family="monospace")
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

    def _asse(self, ax, etichette, titolo, soglia):
        ax.set_yticks(range(len(etichette)))
        ax.set_yticklabels(etichette, fontsize=8)
        ax.set_ylim(-0.5, len(etichette) - 0.5)
        ax.set_title(titolo, fontsize=10, loc="left", color="#e8eaed")
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.axhspan(soglia - 0.5, len(etichette) - 0.5, color="#f28b82", alpha=0.07)
        ax.axhline(soglia - 0.5, color="#f28b82", alpha=0.35, lw=0.8)

    def _consuma(self):
        while True:
            try:
                linea = self.lettore.righe.get_nowait()
            except queue.Empty:
                return
            if self.clip.linea(linea):
                continue                       # riga di riversamento audio
            if linea.startswith("#DEMO"):
                print(f"[demo] ricevuto {linea}")
                nome = self.demo.prossimo_nome()
                self.demo.richiedi()
                # La banda copre la finestra cieca del firmware: in quei
                # secondi il Pico sente il clic del pulsante ma lo scarta,
                # quindi i grafici non sono da leggere.
                t = time.time() - self.t0
                self.demo_eventi.append((t, t + DURATA_CIECA, nome))
                continue
            try:
                d = json.loads(linea)
            except json.JSONDecodeError:
                if self.console:
                    print(linea)
                continue

            # Nella finestra cieca il firmware marca "cieco":1. Quelle
            # letture non vengono proprio registrate: il Pico sente il clic
            # del pulsante e lo classifica, ma e' un suono che il sistema si
            # e' causato da solo e mostrarlo confonderebbe soltanto.
            if d.get("cieco"):
                continue

            t = time.time() - self.t0
            if "p" in d and len(d["p"]) == len(CLASSI_CNN):
                # riordinate secondo la rilevanza dell'asse
                self.prob.append((t, [float(d["p"][i]) for i in PERM]))
            elif d.get("t1") == "Silenzio":
                self.prob.append((t, [0.0] * len(RILEVANZA)))
            if d.get("t1") in LINEA:
                self.vinc.append((t, LINEA.index(d["t1"])))
            if d.get("stato") in STATI:
                self.stato.append((t, STATI.index(d["stato"])))
            for k in ("dsp_us", "dsp_max_us", "inf_us", "adc_ovr", "drop", "clip_perse"):
                if k in d:
                    self.ultimo[k] = d[k]
            if self.console:
                print(linea)

    def _taglia(self, ora):
        limite = ora - self.finestra
        for dq in (self.prob, self.vinc, self.stato):
            while dq and dq[0][0] < limite:
                dq.popleft()

    @staticmethod
    def _agg(punti, passo):
        """Un punto per blocco, con l'indice PIU' ALTO. Con assi ordinati per
        rilevanza il massimo e' l'evento piu' grave: un singolo sparo dentro
        cinque secondi di traffico non deve essere mediato via."""
        if not punti:
            return [], []
        xs, ys, b_corr, mx = [], [], None, None
        for t, v in punti:
            b = int(t // passo)
            if b != b_corr:
                if b_corr is not None:
                    xs.append(b_corr * passo)
                    ys.append(mx)
                b_corr, mx = b, v
            else:
                mx = max(mx, v)
        xs.append(b_corr * passo)
        ys.append(mx)
        return xs, ys

    def aggiorna(self, _):
        self._consuma()
        ora = time.time() - self.t0
        self._taglia(ora)

        inizio_g = max(0.0, ora - self.finestra)
        self.griglia[:] = 0.0
        for t, vec in self.prob:
            col = int((t - inizio_g) / PASSO_GRIGLIA)
            if 0 <= col < self.n_col:
                for r in range(len(RILEVANZA)):
                    if vec[r] > self.griglia[r, col]:
                        self.griglia[r, col] = vec[r]
        self.img.set_data(self.griglia)
        self.img.set_extent([inizio_g, inizio_g + self.finestra,
                             -0.5, len(RILEVANZA) - 0.5])

        if self.vinc:
            self.p_fine.set_data([t for t, _ in self.vinc], [v for _, v in self.vinc])
            self.l_vinc.set_data(*self._agg(self.vinc, self.aggrega_classe))
        if self.stato:
            self.l_stato.set_data(*self._agg(self.stato, self.aggrega))

        # Bande verticali degli invii: stessa colonna su tutti e tre i grafici,
        # cosi' si vede a colpo d'occhio quale tratto e' da ignorare.
        while self.demo_eventi and self.demo_eventi[0][1] < ora - self.finestra:
            self.demo_eventi.popleft()
        for a in self.bande:
            a.remove()
        for t in self.testi:
            t.remove()
        self.bande, self.testi = [], []
        for t0, t1, nome in self.demo_eventi:
            for ax in (self.ax1, self.axL, self.ax2):
                self.bande.append(ax.axvspan(t0, t1, color="#fdd663", alpha=0.16,
                                             zorder=0))
            self.testi.append(self.ax1.text(
                t0, len(RILEVANZA) - 0.35, f" {nome}", fontsize=7.5,
                color="#fdd663", rotation=90, va="top", ha="left", zorder=5))

        for ax in (self.ax1, self.axL, self.ax2):
            ax.set_xlim(inizio_g, max(self.finestra, ora))
            ax.set_xticks(range(int(inizio_g // 5) * 5, int(ora) + 6, 5))

        u = self.ultimo
        parti = [self.lettore.stato]
        if "dsp_us" in u:
            d = f"DSP {u['dsp_us']/1000:.1f} ms"
            if "dsp_max_us" in u:
                d += f" (max {u['dsp_max_us']/1000:.1f})"
            parti.append(d)
            parti.append(f"inferenza {u.get('inf_us',0)/1000:.0f} ms")
            parti.append(f"overrun {u.get('adc_ovr',0)}")
            parti.append(f"scartate {u.get('drop',0)}")
        if self.demo.corrente:
            parti.append(f"DEMO: {self.demo.corrente}")
        parti.append(f"clip salvate {self.clip.salvate}")
        if "clip_perse" in u:
            parti.append(f"perse {u['clip_perse']}")
        if self.clip.dentro:
            parti.append("RICEZIONE CLIP...")
        self.testo.set_text("      ".join(parti))
        return (self.img, self.p_fine, self.l_vinc, self.l_stato,
                self.testo, *self.bande, *self.testi)

    def avvia(self):
        self.ani = FuncAnimation(self.fig, self.aggiorna, interval=250,
                                 blit=False, cache_frame_data=False)
        plt.show()


def main():
    global CARTELLA_DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--finestra", type=float, default=60.0)
    ap.add_argument("--aggrega", type=float, default=5.0,
                    help="blocco di aggregazione per la macchina a stati")
    ap.add_argument("--aggrega-classe", type=float, default=2.0,
                    help="blocco di aggregazione per la classe vincente")
    ap.add_argument("--console", action="store_true")
    ap.add_argument("--campioni", default=CARTELLA_DEMO,
                    help="cartella dei campioni per il pulsante di dimostrazione")
    ap.add_argument("--ritardo", type=float, default=None,
                    help="secondi di attesa fra il clic e il campione "
                         f"(default: durata cieca + {MARGINE_DEMO:g} s)")
    args = ap.parse_args()
    CARTELLA_DEMO = args.campioni
    print("Cruscotto avviato. Le clip finiscono in '%s/'." % CARTELLA_CLIP)
    print("Pulsante di dimostrazione: campioni letti da '%s/'." % CARTELLA_DEMO)
    print("La porta seriale la puo' aprire un solo programma: chiudi read_pico.py.\n")
    ritardo = (DURATA_CIECA + MARGINE_DEMO) if args.ritardo is None else args.ritardo
    print(f"Il campione parte {ritardo:.1f} s dopo il clic, "
          f"cosi' l'attacco cade fuori dalla finestra cieca.")
    Cruscotto(args.finestra, args.aggrega, args.aggrega_classe,
              args.console, ritardo).avvia()


if __name__ == "__main__":
    main()