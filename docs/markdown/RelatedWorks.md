## Related work & useful tools and technologies

Lucrări relevante pentru identificarea cancerului de sân în imagini medicale. Pentru fiecare sunt descrise: datele folosite, algoritmii de AI, performanța obținută, librăriile/tehnologiile folosite și, acolo unde există, linkul către codul public (Git).

---

### 1. Deep Learning to Improve Breast Cancer Detection on Screening Mammography (Shen et al., 2019)

*Articol, Scientific Reports, 9, 12495.* Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802

- **Ce date s-au folosit:** CBIS-DDSM (mamografii digitizate de pe film) pentru antrenare și testare, INbreast (mamografii digitale) pentru validare.
- **Ce algoritmi de AI s-au folosit:** rețea complet convoluțională, antrenată „end-to-end". Adnotările leziunilor sunt necesare doar în prima etapă de antrenare, apoi clasificatorul pe imaginea întreagă folosește doar eticheta imaginii (cancer / fără cancer).
- **Ce performanțe s-au obținut:**
  - CBIS-DDSM: AUC 0,88 pentru cel mai bun model și 0,91 prin medierea a patru modele (sensibilitate 86,1%, specificitate 80,1%).
  - INbreast: AUC 0,95 pentru cel mai bun model și 0,98 cu patru modele (sensibilitate 86,7%, specificitate 96,1%).
- **Ce librării/tehnologii s-au folosit:** Python cu Keras/TensorFlow (de confirmat în README-ul repository-ului; un issue din repository menționează o versiune care rulează pe Python 3 și TF 2).
- **Git:** https://github.com/lishen/end2end-all-conv

---

### 2. Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening (Wu et al., 2020, NYU)

*Articol, IEEE Transactions on Medical Imaging, 39(4), 1184–1194.* Link: https://arxiv.org/abs/1903.08297

- **Ce date s-au folosit:** peste 200.000 de examene de screening (peste 1.000.000 de imagini), cu etichete confirmate prin biopsie.
- **Ce algoritmi de AI s-au folosit:** CNN cu arhitectură în două etape. O rețea de capacitate mare lucrează la nivel de patch, antrenată cu etichete la nivel de pixel, alături de o rețea care învață din etichete la nivel de sân. Se folosește și preantrenare pe clasificarea BI-RADS.
- **Ce performanțe s-au obținut:** AUC 0,895 la predicția prezenței cancerului pe populația de screening. Într-un studiu cu 14 radiologi (câte 720 de examene fiecare), modelul este la nivelul radiologilor experimentați. Un model hibrid (media dintre predicția radiologului și a rețelei) este mai precis decât fiecare separat.
- **Ce librării/tehnologii s-au folosit:** PyTorch (modelul care generează heatmap-uri și modelele de clasificare), plus o implementare TensorFlow a modelului la nivel de imagine.
- **Git:** https://github.com/nyukat/breast_cancer_classifier (modele preantrenate publice)

---

### 3. MammoViT: A Custom Vision Transformer Architecture for Accurate BIRADS Classification in Mammogram Analysis (Al Mansour et al., 2025)

*Articol, Diagnostics, 15(3), 285.* Link: https://www.mdpi.com/2075-4418/15/3/285

- **Ce date s-au folosit:** KAU-BCMD (King Abdulaziz University Mammogram Dataset): 1.416 cazuri de pacienți și 5.662 de mamografii în format DICOM, adnotate de trei radiologi. Clasele BI-RADS folosite sunt 1, 3, 4 și 5, puternic dezechilibrate. Setul este public pe Kaggle: https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset
- **Ce algoritmi de AI s-au folosit:** pipeline în mai mulți pași:
  1. ResNet50 preantrenat pe ImageNet, ca extractor de trăsături (imagini redimensionate la 224 × 224);
  2. SMOTE, pentru echilibrarea claselor;
  3. Vision Transformer, cu hiperparametri optimizați prin Keras Tuner (RandomSearch) și validare încrucișată în 5 pliuri.
- **Ce performanțe s-au obținut:** acuratețe de 97,4% la clasificarea în 4 categorii BI-RADS.
- **Ce librării/tehnologii s-au folosit:** TensorFlow, Keras, Keras Tuner, NumPy, Pandas, scikit-learn (SMOTE, StandardScaler).

---

### 4. Enhancing Breast Cancer Detection with Vision Transformers and Graph Neural Networks (Cai et al., 2025)

*Articol preprint, arXiv:2507.13372.* Link: https://arxiv.org/abs/2507.13372

- **Ce date s-au folosit:** CBIS-DDSM, 10.239 de imagini mamografice (normale și anormale), împărțite 70% antrenare, 10% validare și 20% test. Imaginile sunt redimensionate la 224 × 224, normalizate, cu egalizare de histogramă și augmentare (flip, rotații de ±15°, scalare).
- **Ce algoritmi de AI s-au folosit:** model hibrid ViT + GNN:
  - un ViT-B/16 preantrenat pe ImageNet-21k extrage trăsături globale din întreaga imagine;
  - imaginea este împărțită în patru cadrane, care devin cele patru noduri ale unui graf complet conectat, procesat de două straturi GCN;
  - trăsăturile globale și cele din graf sunt concatenate și fuzionate cu atenție multi-head, apoi un strat final clasifică normal vs. anormal;
  - antrenare cu o funcție de cost contrastivă (inspirată de InfoNCE) și optimizatorul AdamW.
- **Ce performanțe s-au obținut:** acuratețe 84,2%, precizie 83,0%, recall 85,6%, F1 0,84, peste ViT-B/16 (80,3%), DenseNet-121 (79,2%), EfficientNet-B0 (81,0%), Inception-v3 (80,8%) și YOLOv5 (82,1%). În studiul de ablație, ViT singur are 81,5%, iar GNN singur 79,8%.
- **Ce librării/tehnologii s-au folosit:** PyTorch și Deep Graph Library (DGL) pentru partea de GNN, antrenare pe GPU NVIDIA RTX 3090 (batch size 32, 20 de epoci).

---

### 5. A Data Set and Deep Learning Algorithm for the Detection of Masses and Architectural Distortions in Digital Breast Tomosynthesis Images (Buda et al., 2021)

*Articol, JAMA Network Open, și set de date public.* Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8369362

*DBT (Digital Breast Tomosynthesis, tomosinteză digitală a sânului) este un fel de mamografie 3D: în loc de o singură imagine 2D, se obține un volum, adică o stivă de „felii" prin sân.*

- **Ce date s-au folosit:** 22.032 de volume DBT reconstruite, din 5.610 examinări ale 5.060 de paciente, cu mase și distorsiuni arhitecturale adnotate de doi radiologi. Setul este publicat pe The Cancer Imaging Archive.
- **Ce algoritmi de AI s-au folosit:** model de detecție într-o singură fază (single-phase), propus ca punct de plecare (baseline) pentru cercetări viitoare.
- **Ce performanțe s-au obținut:** sensibilitate de 65% la 2 falși pozitivi per volum DBT, pe un set de test de 460 de examene de la 418 paciente. Într-o competiție organizată ulterior pe același set, cu 8 echipe, cea mai bună (NYU B-Team) a obținut o sensibilitate medie de 0,957 pe leziunile biopsiate.
- **Ce librării/tehnologii s-au folosit:** nu am găsit lista în sursele consultate

