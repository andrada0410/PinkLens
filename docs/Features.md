# PinkLens

Prototip de cercetare pentru identificarea cancerului de sân în mamografii. Aplicația este construită cu Streamlit și permite încărcarea unei imagini DICOM, vizualizarea ei și rularea unei clasificări experimentale (malign / benign) realizate de un model AI.

---

## Rulare

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Funcționalități

### 1. Încărcarea imaginii

- Utilizatorul încarcă o mamografie în format **DICOM** (`.dcm` sau `.dicom`).
- Până la încărcarea unui fișier, aplicația afișează un mesaj care invită la încărcare și nu arată restul interfeței.
- Dacă fișierul nu poate fi citit, aplicația afișează un mesaj de eroare cu motivul și se oprește fără să se blocheze.

### 2. Citirea și pregătirea imaginii DICOM

Funcția `load_dicom` citește fișierul cu `pydicom` și transformă datele brute într-o imagine afișabilă:

- **Rescale:** aplică `RescaleSlope` și `RescaleIntercept`, dacă există în fișier.
- **Windowing:** folosește `WindowCenter` și `WindowWidth` pentru contrast și luminozitate. Dacă lipsesc, folosește percentilele 0,5 și 99,5 ale valorilor pixelilor.
- **MONOCHROME1:** imaginile cu valori inversate sunt corectate automat.
- **Fișiere cu mai multe cadre** (ex. tomosinteză): se folosește cadrul din mijloc.
- Rezultatul este o imagine `uint8` în tonuri de gri, folosită atât pentru afișare, cât și ca intrare pentru model.

### 3. Detalii imagine

Sub titlul „Detalii imagine" apar metadatele citite din fișier. Se afișează doar câmpurile care există efectiv în DICOM:

| Câmp afișat | Tag DICOM |
|---|---|
| Lateralitate | `ImageLaterality` / `Laterality` |
| Proiecție | `ViewPosition` |
| Modalitate | `Modality` |
| Dimensiune | calculată din imagine |
| Vârstă pacient | `PatientAge` |
| Data studiului | `StudyDate` |

Fișierele DICOM din multe seturi de date publice sunt anonimizate sau convertite și au puține tag-uri completate (de exemplu `Modality: OT`, adică „Other"). De aceea apar adesea doar Modalitatea și Dimensiunea. Numele și ID-ul pacientului nu sunt afișate intenționat, fiind date personale.

### 4. Analiza cu AI

- Butonul **Analizează** rulează funcția `predict(image)`.
- Rezultatul apare ca un card colorat:
  - roz pal: **Suspect de malign**
  - verde pal: **Probabil benign**
- Sub card sunt afișate probabilitățile pe clase, cu bare de progres.
- Se afișează o recomandare:
  - pentru cazurile suspecte: evaluare suplimentară de către un radiolog (ex. biopsie, ecografie, RMN);
  - pentru celelalte: un mesaj că nu au fost detectate semne suspecte la pragul ales.

### 5. Pragul de decizie

În sidebar există un slider pentru **pragul de decizie** (0,05 – 0,95, implicit 0,50). Un caz este marcat ca malign dacă probabilitatea de malign este mai mare sau egală cu pragul. Pragul poate fi ajustat pentru a compara comportamentul modelului la sensibilități diferite.

### 6. Interfața

- Temă roz pastel, definită în `.streamlit/config.toml` și completată cu CSS în `app.py`.
- Sidebar cu o scurtă descriere a aplicației, slider-ul pentru prag și, în partea de jos, nota „Instrument de cercetare. Rezultatul nu înlocuiește evaluarea unui medic radiolog."
- Layout pe două coloane: imaginea mamografică în stânga, detaliile și rezultatul în dreapta.