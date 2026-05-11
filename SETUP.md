# Setup

## Requisitos

- **OS**: Windows 11 con WSL2 (Ubuntu 24.04) o Linux nativo
- **GPU**: NVIDIA con Compute Capability >= 7.0 (probado en RTX 4070 Laptop, CC 8.9)
- **Driver NVIDIA**: >= 525
- **Python**: 3.10–3.12

> **Nota**: TensorFlow >= 2.11 no soporta GPU en Windows nativo. La ruta probada es WSL2. En Linux nativo basta con saltarse la sección WSL2.

## 1. Preparar WSL2 (solo Windows)

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia, crea usuario, y dentro de Ubuntu:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3-pip build-essential
```

## 2. Entorno virtual

```bash
python3.12 -m venv ~/timnet-env
source ~/timnet-env/bin/activate
pip install --upgrade pip
```

## 3. Clonar e instalar

```bash
git clone <URL-del-repo-privado> timnet-thesis
cd timnet-thesis
pip install -r requirements.txt
```

El paquete `tensorflow[and-cuda]` instala CUDA/cuDNN vía pip — no requiere CUDA Toolkit del sistema.

## 4. Verificar GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Debe imprimir al menos un `PhysicalDevice(device_type='GPU')`. Si aparece vacío, revisa el driver NVIDIA en Windows (no en WSL).

## 5. Preparar datasets

Los archivos `MFCC/*.npy` no vienen en el repo. Para regenerarlos:

1. Descarga los datasets originales desde sus fuentes oficiales:
   - **EMODB**: http://emodb.bilderbar.info/
   - **RAVDESS**: https://zenodo.org/record/1188976
   - **SAVEE**: http://kahlan.eps.surrey.ac.uk/savee/
   - **EMOVO**: http://voice.fub.it/activities/corpora/emovo/
   - **CASIA**: requiere licencia institucional
   - **IEMOCAP**: solicitud a USC (https://sail.usc.edu/iemocap/)

2. Organiza los `.wav` en la estructura esperada por `extract_feature.py`:
   ```
   SER_WAV_DATA/
   ├── EMODB/
   │   ├── angry/*.wav
   │   ├── boredom/*.wav
   │   └── ...
   └── ...
   ```

3. Extrae features MFCC:
   ```bash
   python extract_feature.py --data_name EMODB
   ```
   Esto genera `MFCC/EMODB.npy`.

## 6. Ejecutar

```bash
# Baseline
python main.py --mode train --data EMODB --epoch 500

# V-A Fusion
python main.py --mode va_train --data EMODB --epoch 500
```

## Tiempos de referencia (RTX 4070 Laptop, 5.5 GB VRAM)

| Dataset | Epochs | Tiempo aprox. |
|---------|:------:|:-------------:|
| EMODB   | 500    | ~2 h          |
| IEMOCAP | 500    | ~6 h (dilation_size=10) |

## Problemas conocidos

- **OneDrive + h5py**: si el proyecto vive en una ruta sincronizada con OneDrive, el guardado de pesos puede fallar. Usa `--model_path /tmp/Models/` para escribir en ruta local.
- **WSL2 y sudo**: Ubuntu 24.04 aplica PEP 668. Instala todo dentro del venv, nunca global.
- **XLA overhead**: el primer epoch de cada fold añade ~30 s de compilación; normal.
