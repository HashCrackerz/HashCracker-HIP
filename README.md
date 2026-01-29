<p align="left">
  <img width="150" alt="logo" src="https://github.com/user-attachments/assets/73297594-3581-4afd-ad0b-39d4bc0e66bf" />
</p>

# HashCracker (HIP)
## _Parallel SHA-256 Brute Force & Dictionary (salted) Password Cracker_


![alt text](https://img.shields.io/badge/Language-C++%20%7C%20CUDA-green)

![alt text](https://img.shields.io/badge/Platform-AMDA%20(HIP)-red) 

![alt text](https://img.shields.io/badge/Algorithm-SHA256-purple)

Progetto per il corso di Sistemi di Elaborazione Accelerata della facoltà di Ingegneria Informatica Magistrale di UniBo.
Applicazione **parallela** per il cracking di password tramite attacco Brute Force su hash SHA-256 (anche salted) e attacco a dizionario, 
con confronto prestazionale tra implementazione Sequenziale (CPU) e Parallela (HIP per GPU AMD).

## 📝 Descrizione
Il progetto implementa un **cracker di password** che supporta diverse modalità di attacco per invertire hash SHA-256. 
L'obiettivo principale è dimostrare lo speedup ottenibile passando da un'esecuzione seriale su CPU a un'esecuzione 
massivamente parallela su GPU AMD in questo caso.

## ⚙️ Funzionalità
- **Brute Force Incrementale**: Generazione dinamica di password dato un charset e un range di lunghezza (min-max).
- **Attacco a Dizionario**: Supporto per wordlist esterne.
- **Supporto Salt**: Gestione di hash saltati (Brute Force e attacco a dizionario).
- **Multi-Platform**: Codice nativo CUDA per **NVIDIA** e script di porting semi-automatico per **AMD** HIP.

## 📂 Struttura del Progetto
- `Sequenziale/`: Implementazione **sequenziale** di riferimento (usa [OpenSSL](https://openssl-library.org/)).
- `HIP_NAIVE/`: **Prima implementazione GPU** (uso memoria globale).
- `HIPv1/`: Ottimizzazione memoria (uso Constant Memory per charset e target).
- `HIPv2/`: Ottimizzazione kernel (loop unrolling, register optimization per SHA-256).
- `UTILS/`: Funzioni di supporto (I/O file, parsing argomenti).
- `SHA256_HIP/`: implementazione **HIP per SHA256**, basata sull'implementazione di [mochimodev](https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu).
- `SHA256_HIP_OPT/`: implementazione **HIP ottimizzata per SHA256** (usato da HIPv2 ed estensione).
- `ESTENSIONE/`: contiene l'implementazione delle'estensione del progetto, ossia l'attacco a dizionario e l'hash cracking con salt.
- `convert_to_hip.ps1`: Script PowerShell per il **porting semi-automatico** su AMD ROCm/HIP (e guida alle rimanenti operazioni per il porting).
- `kernel_[versione_progetto].cu`: file per eseguire la corrispondente versione.
Tutte le versioni HIP[versione_progetto] (eseguite dai rispettivi file kernel) hanno come dipendenza i file di
`UTILS` e `SHA256_HIP`, ad eccezione della v2 che usa `SHA256_HIP_OPT` (versione ottimizzata) invece di `SHA256_HIP`.
Nota che l'implementazione HIP é sostanzialmente analoga a quella CUDA riportata nella repository [HashCracker-CUDA](https://github.com/HashCrackerz/HashCracker-CUDA).

Llo script di porting si occupa di tradurre opportunamente le funzioni, i tipi e importare o rimuovere le librerie, oltre che
convertire ad esempio gli import con i doppo apici in parentesi angolari. I file vengono sovrascritti e rinominati, ma la
struttura progettuale resta la medesima. I file di questa repository sono i risultato proprio di questa conversione e di un
porting manuale rifinito (aggiunta di extern C ai kernel, correzione spaziatura invocazione thread e poco altro). 

## 🛠️ Requisiti
- Hardware:
  - **GPU AMD**

- Software:
  - **OpenSSL** (per l'implementazione CPU)
  - **Compilatore C++** (MSVC su Windows, GCC/Clang su Linux)
  - **PowerShell** (per lo script di porting)
  - **ROCm** (per compilare con hipcc e per l'ambiente AMD).
 
## 🚀 Compilazione

### AMD HIP (Porting)
Il progetto include uno script per convertire automaticamente il codice CUDA in HIP.
1. Eseguire lo script di conversione:
   ```powershell
   .\convert_to_hip.ps1
   ```
2. Controllo manuale
   1. Tutti i kernel devono essere lanciati con la sintassi con la tripla parentesi angolare senza spazi (esempio `kernel <<< block, thread >>> (args)`).
   2. Le funzioni kernel devono essere `__global__` ed `extern "C" {}` (sia in .cu che in .cuh).
   3. Controllare gli errori di compilazione.
2. Compilazione con hipcc (esempio per Windows):
   ```powershell
      hipcc -fgpu-rdc -O3 -std=c++14 --offload-arch=native `
        kernel_estensione.cu `
        ESTENSIONE/DIZIONARIO/hip_dizionario.cu `
        ESTENSIONE/SALT/hip_salt.cu `
        SHA256_HIP/sha256.cu `
        UTILS/hip_utils.cu `
        UTILS/utils.cpp `
        -I. -I./ESTENSIONE/DIZIONARIO -I./ESTENSIONE/SALT -I./SHA256_HIP -I./UTILS `
        -I"D:\OpenSSL-Win64\include" `
        -L"D:\OpenSSL-Win64\lib\VC\x64\MTd" `
        -l libcrypto.lib `
        -D_CRT_SECURE_NO_WARNINGS `
        -o estensione_amd.exe
   ```
   _(cambiare i nomi dei file e delle dipendenze in base alla versione da compilare)_

## 💻 Utilizzo
Il programma accetta i parametri da riga di comando per la massima flessibilità:
```cmd
./brute_force_cuda [<blockSize>] <hash_target> <min_len> <max_len> <file_charset> [<salt> <dizionario-si/no> <file_dizionario>]
```
La `blockSize` va passata sempre e solo negli script paralleli su gpu (sia CUDA che HIP). \
Il dizionario (flag e file path) e salt vanno passati solo negli script estensione. \
_Nota_: nella versione estensione `max_len` comprende la lunghezza del salt. 

Esempio: \
Cercare la password dell'hash (corrispondente a "qwerty") con lunghezza 6, usando il charset standard:
```cmd
./brute_force_cuda 256 qwerty 1 6 ASSETS/CharSet.txt
```


## 📊 Analisi delle Performance
I test sono stati condotti su:
- desktop con Ryzen 9 9900X e RX 9070XT

### Approfondimento Tecnico: Analisi
L'algoritmo SHA-256 è fortemente **Compute-Bound**. L'implementazione v2 utilizza pesantemente i registri per mantenere lo stato 
dell'hash ed evitare latenze di memoria locale/globale. Sebbene l'alto numero di registri (118) limiti il numero di warp attivi 
(bassa occupancy), la velocità di esecuzione del singolo thread aumenta drasticamente. In questo scenario, massimizzare l'IPC 
(Instructions Per Cycle) si è rivelato più efficace che massimizzare il parallelismo a livello di latenza (Occupancy).

Inoltre, l'utilizzo di dimensioni del blocco minori (64/128 thread) ha portato a performance migliori rispetto ai classici 256, 
grazie a una migliore gestione della Tail Effect (quantizzazione delle wave) e minore overhead di scheduling.

L'implementazione dell'estensione ha sostanzialmente le stesse prestazioni della v2 (in quanto utilizza praticamente il metesimo codice), 
con l'aggiunta che per l'attacco a dizionario, il tempo in caso di hit é certamente inferiore a testare tutte le combinazioni.

## 👥 Autori
- [Andrea Vitale](https://github.com/WHYUBM)
- [Matteo Fontolan](https://github.com/itsjustwhitee)

## 📜 Licenza
Questo progetto è distribuito sotto licenza AGPL. Vedi il file `LICENSE` per i dettagli.
