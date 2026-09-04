> Smart City Audio — classificatore acustico in tempo reale su Raspberry Pi Pico.
> 

<aside>
💡

#### Definizioni:

- **ADC (Analog to Digital Converter)**: Hardware che campiona il voltaggio analogico del microfono e lo traduce in valori digitali.
- **DMA (Direct Memory Access)**: Sottosistema che sposta flussi di byte dai convertitori alla memoria senza rubare cicli di clock al processore.
- **DSP (Digital Signal Processing)**: L'elaborazione che manipola i segnali (filtri, trasformate) per estrarre le frequenze dal suono.
- **FFT (Fast Fourier Transform)**: Operazione matematica che isola le singole frequenze che compongono l'onda acustica.
- **SDK (Software Development Kit)**: Insieme di strumenti preconfezionati che i produttori di hardware forniscono.
- **FIR (Finite Impulse Response)**: Algoritmo matematico per isolare e tagliare specifiche frequenze.
- **Bande Mel (Melody)**: Intervalli di frequenza basati sulla scala Mel, una scala acustica che imita il modo in cui l'orecchio umano percepisce l'altezza dei suoni.
</aside>

## Il caso di Trento

Il progetto nasce da un caso concreto: nel 2024 il Garante per la protezione dei dati personali è intervenuto su un progetto di sorveglianza acustica del Comune di Trento.
Il problema non era l'idea di monitorare i suoni della città — era che **i microfoni trasmettevano audio in chiaro**, dal quale si potevano riconoscere voci e conversazioni. Dati personali, quindi, trattati senza le garanzie che il **GDPR** richiede. La domanda da cui sono partito è questa: si può ottenere lo stesso risultato senza che l'audio esca mai dal dispositivo? La risposta è sì, se la classificazione avviene a bordo. E il vincolo interessante, per questo corso, è che deve avvenire in tempo reale su un microcontrollore.

## 1 - Il problema

Nell'approccio tradizionale il microfono è un sensore stupido: cattura e spedisce, e tutta l'intelligenza sta sul server. Questo significa che l'audio grezzo attraversa la rete, viene archiviato, e chiunque vi acceda può ascoltare conversazioni private.
Nell'approccio che ho adottato il dispositivo è autonomo: **elabora lo spettrogramma**, **lo classifica** con una rete neurale, e **trasmette soltanto un'etichetta testuale**. Quello che vedete in fondo è esattamente ciò che esce dalla seriale: una quarantina di byte. Da quella stringa non si può ricostruire nulla dell'audio originale. È una scelta architetturale che risolve il problema alla radice, invece di aggiungere cifratura sopra un dato che non doveva essere raccolto.

## 2 - Hardware

Il Raspberry Pi Pico monta un RP2040 con due core, ma ne ho usato uno solo. Le risorse sono modeste: 264 kilobyte di RAM, di cui il progetto ne occupa 220, e due megabyte di flash. Il vincolo che ha condizionato ogni scelta è però un altro: non ha unità in virgola mobile. Ogni moltiplicazione float viene emulata dal compilatore e costa decine di cicli invece di uno.

## 3 - Sistema operativo

La scelta di usare FreeRTOS come sistema operativo è dovuta al fatto che ci sono task con periodi diversi che devono convivere sullo stesso core, e deve essere permessa la preemption per chi è dotato di maggiore priorità. FreeRTOS ha un porting ufficiale per l'RP2040, e ha un'occupazione contenuta: ho dichiarato 80 kilobyte di heap e ne uso meno di 50. 
Lo scheduler è a priorità fisse con prelazione. L'assegnazione delle priorità segue Rate Monotonic: periodo più breve, priorità più alta.

## 4 - Architettura

La catena ha cinque stadi:

- Il **microfono** collegato all'ADC a dodici bit del Pico.
- L'**acquisizione** avviene a 48 kHz in DMA, con due buffer alternati.
- Il **DSP** filtra, decima a 16 kHz e calcola uno spettrogramma mel a 32 bande.
- La **rete convoluzionale** quantizzata a 8 bit classifica ogni finestra da un secondo in otto categorie.
- Infine una **macchina a stati** deterministica legge la sequenza di predizioni e riconosce gli scenari.

Quello che rende il progetto interessante per questo corso non è nessuno di questi blocchi preso da solo: è che ognuno ha un costo di calcolo misurato e una scadenza da rispettare, e che devono convivere su un solo core.

## 5 - Il vincolo temporale

Il DMA riempie un buffer ogni 32 millisecondi, e i buffer sono due: mentre uno si riempie, l'altro viene elaborato. 

- Se il DSP non finisce prima che arrivi il blocco successivo, il DMA scrive sopra i campioni non ancora letti, e quell'audio è perso — non c'è nessun modo di recuperarlo. Questa è una **scadenza dura**.
- L'inferenza è diversa: se salta un turno, mezzo secondo dopo arriva la finestra successiva e il sistema riprende. Si perde una predizione, non il dato: è una **scadenza morbida**.
- La trasmissione seriale **non ha scadenza**: il messaggio è già in memoria e ci resta finché non parte.

Queste tre righe sono anche l'ordine delle priorità. Chi rischia una perdita irrecuperabile va servito per primo: DSP, poi inferenza, poi seriale.

## 5 - Progetto dei task

I task sono tre:

- **Task_DSP** ha priorità 3, la più alta, e si attiva ogni 32 millisecondi quando il DMA sblocca un semaforo: filtra, decima, calcola FFT e bande mel.
- **Task_ML** ha priorità 2 e si attiva ogni 496 millisecondi: quantizza la finestra, esegue la rete e aggiorna la macchina a stati.
- **Task_Serial** ha priorità 1, la più bassa, ed è sporadico: svuota la coda dei messaggi, riversa le registrazioni di prova e legge i comandi.

Una scelta che vale la pena motivare: la macchina a stati e il pulsante non hanno task propri. Hanno rispettivamente lo stesso periodo dell'ML e DSP e hanno tempi irrisori. Due task dedicati avrebbero aggiunto stack, semafori e cambi di contesto.

## 6 - Occupazione del processore

Questi sono costi misurati sul dispositivo, non stimati. 

- Il DSP impiega 20,8ms ogni 32, cioè due terzi del processore.
- L'inferenza è più delicata da interpretare. Il firmware misura 298ms, ma è tempo trascorso, non tempo lavorato: in quei 298ms il task viene interrotto dal DSP ogni 32ms e resta fermo finché quello non finisce. Il DSP prende il 65% del processore, quindi all'inferenza resta il 35%. In 298ms di orologio ha lavorato per il 35% del tempo, cioè circa 105ms. Su un periodo di 496 sono il 21%. Sommando: 65 più 21 fa 86 per cento di utilizzo, con il 14 per cento libero.

## 7 - Schedulabilità: il criterio

Il primo strumento che si applica è il criterio di Liu e Layland: un insieme di task periodici con priorità Rate Monotonic è schedulabile se:

$$
U \leq n(2^{\frac{1}{n}}-1)
$$

Con tre task quel limite vale 78%. Il mio sistema è all'86%, quindi la condizione non è soddisfatta. Quindi non è possibile stabilirne la schedulabilità.

## 8 - Schedulabilità: RTA

L’analisi RTA calcola il tempo di risposta come somma del proprio tempo di calcolo più l'interferenza dei task a priorità maggiore. L'interferenza non è nota in anticipo, perché dipende da quante volte il task più prioritario si attiva durante la risposta, che a sua volta dipende dalla risposta stessa. Si risolve iterando fino al punto fisso. 

- Per Task_ML: si parte dai suoi 105ms di calcolo, si contano quante attivazioni del DSP ci stanno dentro, si aggiunge la loro interferenza, si ricalcola. La successione converge a 313ms. La scadenza è 496, quindi il sistema è schedulabile.
- Per il DSP il calcolo è immediato: essendo il più prioritario non subisce interferenza, e i suoi 20,8 millisecondi stanno sotto i 32 di periodo.

Vale la pena notare che il valore calcolato, 313 millisecondi, è vicino ai 298 misurati sul dispositivo.

## 9 - Robustezza

Il sistema salva una registrazione di prova quando riconosce un evento pericoloso. Quella clip dura 2s: 32.000 byte che, in base64, diventano 42.700 caratteri. A 115200 baud la trasmissione richiede 3,7s. Attenzione però: in questi 3,7s la CPU lavora effettivamente solo per (circa) 21ms per codificare i dati e riempire il buffer hardware, restando libera per il resto del tempo. Il collo di bottiglia è quindi puramente fisico (il cavo), non computazionale.  Ciononostante, se gli eventi si susseguono, le richieste arrivano più in fretta di quanto il canale seriale le smaltisca: è una condizione di sovraccarico. La scelta di progetto è stata rifiutare il lavoro in eccesso invece di accodarlo, con un tempo morto di dieci secondi e un contatore degli scarti. Il motivo del rifiuto è semplice: accodare significherebbe trasmettere, fra un minuto, un evento avvenuto un minuto prima. Per un sistema di allerta un dato tardivo vale meno di un dato assente. C'è un principio generale dietro: nessuna politica di scheduling risolve un sovraccarico, perché non esiste modo di eseguire più lavoro di quanto ne consenta il tempo fisico della periferica. Lo scheduler decide l'ordine, non crea capacità dal nulla.

## 10 - Riconoscimento

Sulla rete neurale sarò breve, perché non è il centro di questo corso. È una convoluzionale a tre blocchi, quantizzata a 8bit, che occupa 31 kB di flash ed esegue circa 107.000 moltiplicazioni per inferenza. Classifica otto categorie acustiche e raggiunge l'88,9% sul test set. 
Un'osservazione: l'88,9% è misurato su file audio. Registrando con il microfono reale l'accuratezza scende al 42,7%. Il divario è la catena di acquisizione, non il modello, e un riaddestramento degli ultimi strati su registrazioni fatte dal dispositivo lo ha riportato al 65,2%.

## 11 - Macchina a stati

Sopra la rete neurale c'è uno strato che ragiona sul tempo. Serve perché alcuni eventi non esistono dentro una finestra da un secondo. Uno sparo e un fuoco d'artificio sono acusticamente identici: li distingue solo la ripetizione. 
Un incidente non è un suono ma una sequenza — stridio di gomme, impatto, vetri. E ci sono precondizioni che nessuna rete dedurrebbe da sola: se un attimo prima dell'urto la scena era deserta — niente veicoli, niente persone — quell'impatto è semplicemente un rumore forte. 
Avevo provato a farlo imparare a una seconda rete addestrata su sequenze sintetiche: si fermava al 76%, con risultati scadenti proprio sugli eventi puntuali. La macchina a stati deterministica fa meglio perché una regola categorica conviene scriverla, non sperare che la rete la deduca da ottocento esempi. Costa decine di microsecondi per ciclo e gira dentro Task_ML.

## 12 - Il pulsante di dimostrazione

Un'ultima aggiunta, pensata per la dimostrazione dal vivo: un pulsante sulla breadboard che fa riprodurre al PC un campione audio, così potete vedere il riconoscimento in tempo reale. La domanda è: è stato creato un nuovo task? No, e la scelta è motivata. Un task si giustifica quando ha un periodo proprio o quando può bloccare gli altri; un pulsante non ha né l'uno né l'altro. La lettura sta dentro Task_DSP, e questo è il punto che conta: quel task ha periodo fisso di 32ms e non si blocca mai, quindi il campionamento del pin è deterministico. C'è anche un vantaggio collaterale: l'antirimbalzo non richiede un timer, perché tre letture consecutive a 32ms sono già 96ms di contatto stabile. La periodicità del task fa da base dei tempi. Sulla schedulabilità l'effetto è nullo: due letture di un registro GPIO costano decine di nanosecondi su un budget di 32ms, quindi il tempo di risposta calcolato prima resta valido. 

Infine, quando il pulsante viene premuto il firmware ignora le tre inferenze successive: il clic meccanico arriva al microfono e verrebbe classificato come uno scoppio. Non è un falso positivo da tarare con le soglie, è un evento che il sistema stesso ha causato e di cui conosce l'istante esatto.

## 13 - Bilancio Finale

Il bilancio onesto. Dal punto di vista del tempo reale gli obiettivi sono raggiunti: il sistema è schedulabile, dimostrato analiticamente. E la verifica sperimentale lo conferma: in ore di funzionamento i contatori di overrun dell'ADC e di finestre scartate sono rimasti a zero. Quei contatori vengono trasmessi in ogni messaggio, quindi il rispetto delle scadenze è monitorato in continuazione, non verificato una volta e dimenticato. L'obiettivo di privacy è raggiunto per costruzione, perché l'audio non lascia mai il dispositivo. 

Il limite principale è l'accuratezza reale: il 65% contro l'89 sul dataset. Il collo di bottiglia non è il modello né il processore, è la catena analogica — un elettrete su un ADC a dodici bit con escursione limitata perde le componenti in alta frequenza, che sono proprio quelle che distinguono vetri, spari e impatti. Alcune classi, come ambiente urbano e veicoli, restano poco separabili anche in condizioni ideali.

### Impossibilità di Unbounded Priority Inversion

- Assenza di mutua esclusione: Il sistema non utilizza costrutti bloccanti (come i mutex) per proteggere la memoria condivisa. Nessun task a bassa priorità ha il potere di "tenere in ostaggio" una risorsa indispensabile a un task più urgente.
- Separazione Spaziale (Double Buffering): Nel caso dell'acquisizione, la memoria è fisicamente sdoppiata. Mentre il produttore (l'hardware) riempie il buffer A, il consumatore (il task) svuota il buffer B. Al ciclo successivo, i ruoli si invertono. Non essendoci mai sovrapposizione sulla stessa area di memoria, la sezione critica semplicemente non esiste.
- Separazione Temporale (Handoff stretto): Quando i task si passano i dati elaborati, lo fanno in modo strettamente sequenziale. Il task produttore riempie il buffer, invia il segnale di "pronto" al consumatore e smette di toccare quell'area finché il consumatore non ha finito. Se nel frattempo arrivano nuovi dati, il produttore preferisce scartarli (drop) pur di non sovrascrivere la memoria che l'altro task sta leggendo in quel momento.

### Acquisizione

L'acquisizione è il percorso critico. L'ADC lavora in modo continuo a 48 kHz esatti, ottenuti con un divisore intero, e il DMA scrive in due buffer alternati. Mentre uno si riempie, il task elabora l'altro: la CPU non aspetta mai il convertitore. Quando un blocco è completo, la routine di interruzione riarma immediatamente il canale sull'altro buffer e sblocca il task tramite un semaforo.
