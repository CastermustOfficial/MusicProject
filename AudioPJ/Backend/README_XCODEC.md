# XCodec Real Audio Decoder

## Current Status

Il sistema ora ha **due modalità di decodifica**:

### 1. **Placeholder Decoder** (Attivo ora)
- ✅ Funziona subito senza installazioni
- ⚠️ Genera suoni elettronici basati sui token
- ⚠️ NON è musica reale

### 2. **Real XCodec Decoder** (Da attivare)
- 🎵 Genera musica VERA dai token
- ✅ Usa il modello XCodec ufficiale
- 📦 Richiede installazione pacchetti aggiuntivi

## Come Attivare il Decoder Reale

### Step 1: Installa le dipendenze

```bash
pip install -r requirements_xcodec.txt
```

Questo installerà:
- `transformers` - Per caricare il modello XCodec da Hugging Face
- `accelerate` - Per inferenza efficiente
- `scipy` - Per resampling audio
- `librosa` - Per processing audio avanzato

### Step 2: Riavvia uvicorn

```bash
uvicorn backend.main:app --reload
```

### Step 3: Verifica nel log

Quando uvicorn si avvia, dovresti vedere:

```
INFO: Real XCodec decoder available
```

Invece di:

```
WARNING: Real XCodec decoder not available
```

### Step 4: Genera una canzone

Quando generi, vedrai:

```
🎵 Using real XCodec decoder for high-quality audio...
```

## Come Funziona

### Pipeline Completa:

1. **Stage 1 (YuE-s1-7B GGUF)**
   - Genera lyrics + token audio testuali
   - Output: `<xcodec/0/464><xcodec/0/130>...`

2. **Token Extraction**
   - Estrae i numeri dai token: `[464, 130, ...]`

3. **XCodec Decoder** (Nuovo!)
   - Carica modello `HKUSTAudio/xcodec2` da Hugging Face
   - Decodifica i token in forma d'onda audio reale
   - Output: musica vera!

4. **Post-processing**
   - Resampling a 44.1kHz se necessario
   - Padding/trimming a 30 secondi
   - Normalizzazione

## Modelli Supportati

Il decoder prova nell'ordine:

1. **XCodec2** (`HKUSTAudio/xcodec2`) - Preferito
   - Più recente e semplice
   - Codebook singolo
   - Migliore qualità

2. **XCodec** (`facebook/xcodec`) - Fallback
   - Originale
   - Multi-level RVQ

## Requisiti Hardware per Decoder Reale

### GPU (Raccomandato)
- VRAM: ~2-4GB per XCodec
- Inferenza veloce (~5-10 secondi)

### CPU (Funziona ma lento)
- RAM: 8GB+
- Inferenza lenta (~30-60 secondi)

## Troubleshooting

### "Real XCodec decoder not available"
**Causa:** Pacchetti non installati o errore import

**Soluzione:**
```bash
pip install transformers accelerate scipy librosa
```

### "XCodec model not available"
**Causa:** Download modello fallito

**Soluzione:**
1. Controlla connessione internet
2. Prova manualmente:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("HKUSTAudio/xcodec2", trust_remote_code=True)
```

### "Out of memory" durante decodifica
**Causa:** Token troppi per VRAM disponibile

**Soluzione:**
1. Usa CPU mode (automatico se no GPU)
2. Riduci durata generazione Stage 1

## Differenza Placeholder vs Real

### Placeholder (Attuale)
```
Token: 464 → Freq: 420Hz
Token: 928 → Freq: 880Hz
Token: 130 → Freq: 250Hz
```
= Toni sintetici modulati

### Real XCodec
```
Token: 464 → Audio Feature Vector
Token: 928 → Audio Feature Vector
Token: 130 → Audio Feature Vector
```
→ Decoder neurale → **Musica reale!**

## Performance

| Modalità | Qualità | Velocità | VRAM |
|----------|---------|----------|------|
| Placeholder | ⭐ | ⚡⚡⚡ | 0GB |
| XCodec (GPU) | ⭐⭐⭐⭐⭐ | ⚡⚡ | ~3GB |
| XCodec (CPU) | ⭐⭐⭐⭐⭐ | ⚡ | 0GB |

## Fonti e Riferimenti

- [X-Codec Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/xcodec)
- [X-Codec GitHub](https://github.com/zhenye234/xcodec)
- [XCodec2 Model](https://huggingface.co/HKUSTAudio/xcodec2)
- [YuE Model Paper](https://arxiv.org/html/2503.08638v1)
