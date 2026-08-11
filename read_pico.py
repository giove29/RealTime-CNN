import serial
import serial.tools.list_ports
import json
import time

BAUD_RATE = 115200
PICO_VID = "2E8A"          # ID hardware ufficiale Raspberry Pi


def trova_porta_pico(silenzioso=False):
    """Scansiona le porte USB e restituisce quella del Pico, o None."""
    for porta in serial.tools.list_ports.comports():
        if PICO_VID in porta.hwid:
            if not silenzioso:
                print(f"Pico trovato sulla porta: {porta.device}")
            return porta.device
        if "ttyACM" in porta.device:          # fallback Linux
            if not silenzioso:
                print(f"Pico (probabile) su porta generica: {porta.device}")
            return porta.device
    return None


def attendi_pico(timeout_s=None):
    """Aspetta che il Pico compaia. Serve dopo un riavvio del watchdog:
    la porta USB scompare e viene rienumerata dopo qualche secondo."""
    inizio = time.time()
    primo = True
    while True:
        porta = trova_porta_pico(silenzioso=not primo)
        if porta:
            return porta
        if primo:
            print("In attesa del Pico (riavvio in corso?)...")
            primo = False
        if timeout_s and (time.time() - inizio) > timeout_s:
            return None
        time.sleep(0.5)


def mostra(linea):
    """Stampa una riga: JSON formattato se lo e', testo grezzo altrimenti."""
    orario = time.strftime('%H:%M:%S')
    try:
        d = json.loads(linea)
    except json.JSONDecodeError:
        print(f"[{orario}] {linea}")
        return

    print(f"[{orario}] Rilevamento sonoro:")
    for pos, n in ((" 1", 1), (" 2", 2), (" 3", 3)):
        etichetta = d.get(f"t{n}", "?")
        prob = d.get(f"p{n}", 0.0)
        print(f"   {pos}. {etichetta:<18} {prob * 100:5.1f}%")
    if "scenario" in d:
        print(f"       scenario: {d['scenario']} ({d.get('sp', 0)*100:.0f}%)")
    if "dsp_us" in d:
        print(f"       DSP {d['dsp_us']} us (max {d.get('dsp_max_us','?')})  "
              f"inferenza {d.get('inf_us','?')} us  "
              f"overrun ADC {d.get('adc_ovr','?')}  scartate {d.get('drop','?')}")
    print("-" * 60)


def ascolta():
    print("In ascolto. Ctrl+C per uscire.\n" + "=" * 60)
    while True:
        porta = attendi_pico()
        try:
            with serial.Serial(porta, BAUD_RATE, timeout=1.0) as sp:
                print(f"--- connesso a {porta} ---")
                while True:
                    raw = sp.readline()
                    if not raw:
                        continue                      # solo timeout di lettura
                    linea = raw.decode('utf-8', errors='replace').strip()
                    if linea:
                        mostra(linea)
        except (serial.SerialException, OSError, PermissionError) as e:
            # Il Pico si e' riavviato: la porta sparisce e torna dopo qualche
            # secondo. Non e' un errore, si riprova.
            print(f"\n--- connessione persa ({type(e).__name__}), riconnessione ---\n")
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nAscolto terminato.")
            return


if __name__ == "__main__":
    try:
        ascolta()
    except KeyboardInterrupt:
        print("\nAscolto terminato.")
