# Pink Lens


## 1. Tema proiectului

**Titlul temei:** AI pentru sănătatea femeilor: Identificarea cancerului de sân

**Owner:** dl. dr. Andrei Roman

**Scop:** dezvoltarea unui sistem inteligent care să ajute medicii în diagnosticarea timpurie a cancerului de sân.

**Ideea de bază:** detectarea cancerului de sân în mamografii este o problemă critică în imagistica medicală. Interpretarea mamografiilor este dificilă din cauza variațiilor în densitatea țesuturilor, a structurilor suprapuse și a diferențelor subtile dintre tumorile benigne și cele maligne, iar examinarea manuală de către radiologi consumă timp și este predispusă la erori umane. Problema trebuie abordată cu un algoritm inteligent: pornind de la seturi de date cu mamografii, se folosesc modele de AI bazate pe arhitecturi de tip Transformer sau GNN pentru a identifica tumorile maligne și benigne în imagini. Modelele pot fi pre-antrenate pe alte seturi de date și apoi fine-tunate pe mamografii.

**Proiect:** Pink Lens

---

## 2. Descrierea problemei (varianta plastică)

Imaginează-ți un radiolog care analizează zeci de mamografii pe zi. Fiecare imagine este o „hartă" în tonuri de gri a sânului, în care țesutul dens, vasele de sânge, glandele și eventualele tumori apar toate ca zone mai deschise sau mai întunecate, adesea suprapuse. O tumoră mică, la început de drum, poate arăta aproape identic cu un țesut sănătos dens, iar diferența dintre o formațiune **benignă** (inofensivă) și una **malignă** (periculoasă) este uneori de câțiva pixeli, de o margine ușor neregulată sau de o grupare de micro-calcificări.

Sistemul propus este un „al doilea set de ochi" pentru medic: un model de inteligență artificială care primește mamografia, o analizează și spune:

- dacă există o zonă suspectă,
- unde se află aceasta,
- cât de probabil este să fie malignă.

AI-ul **nu înlocuiește medicul**, ci îl ajută să nu rateze cazurile dificile și să-și petreacă timpul acolo unde contează cel mai mult.

---

## 3. Descrierea problemei (varianta formală)

**Ce este o mamografie.** O mamografie este o matrice de pixeli, cu `H` rânduri și `W` coloane. Fiecare pixel are o valoare numerică de intensitate (de exemplu între 0 și 255, sau între 0 și 65535 la imaginile pe 16 biți): valorile mici corespund zonelor întunecate, iar cele mari zonelor deschise (țesut dens, tumori, calcificări). Imaginea este deci un tabel de numere, iar „a vedea o tumoră" înseamnă a găsi un anumit tipar în acest tabel.

**Ce vrem să aflăm.** Pentru fiecare imagine există un răspuns corect, stabilit de medici (de obicei confirmat prin biopsie):
- `0` pentru benign (sau normal),
- `1` pentru malign.

**Setul de date.** Avem o colecție de `N` exemple, fiecare fiind o pereche _(imagine, răspuns corect)_. Acesta este materialul din care modelul învață. O parte din exemple se păstrează deoparte și nu se folosesc la învățare, ci doar la testarea finală.

**Ce face modelul.** Pentru o imagine nouă, pe care nu a mai văzut-o, modelul calculează un număr `p` între 0 și 1, care spune cât de probabil este ca imaginea să conțină o tumoră malignă:
- `p` aproape de 0 → probabil benign,
- `p` aproape de 1 → probabil malign.

Decizia se ia cu un prag: dacă `p` depășește un prag ales (de exemplu 0,5), imaginea este marcată ca suspectă. Pragul se poate coborî ca să prindem cât mai multe cazuri maligne, cu prețul unor alarme false în plus.

**Cum învață modelul.** Modelul are multe numere interne (parametri), la început alese aproape la întâmplare. Procesul este iterativ:

1. i se arată o imagine din setul de date și produce un `p`;
2. `p` se compară cu răspunsul corect, iar diferența dintre ele este „eroarea";
3. parametrii se ajustează puțin, în direcția în care eroarea scade;
4. se repetă pentru toate imaginile, de mai multe ori.

La final, parametrii sunt cei pentru care modelul greșește cât mai puțin pe exemplele văzute și, sperăm, și pe imaginile noi.

**Dezechilibrul claselor.** Cazurile maligne sunt mult mai rare decât cele benigne sau normale. Dacă, de exemplu, doar 5 imagini din 100 sunt maligne, un model care răspunde mereu „benign" ar avea 95% acuratețe, deși nu ar detecta niciun cancer. De aceea, la antrenare se acordă o importanță mai mare cazurilor maligne, iar la evaluare se urmăresc metrici precum sensibilitatea (secțiunea 8), nu doar acuratețea.

**Variante ale sarcinii.** În funcție de ce cerem modelului, răspunsul poate avea forme diferite:

| Sarcină               | Ce primim ca răspuns                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| Clasificare binară    | benign sau malign                                                                                         |
| Clasificare BI-RADS   | o categorie de la 1 la 5 (cât de suspectă este imaginea)                                                  |
| Detecție / localizare | un chenar în jurul leziunii, plus scorul de încredere                                                     |
| Segmentare            | o mască, adică un tabel de aceeași mărime cu imaginea, cu 1 pe pixelii care aparțin leziunii și 0 în rest |

**Arhitecturi propuse:**

- **Transformere vizuale (ViT, DeiT, CrossViT, MammoViT):** imaginea este împărțită în patch-uri `16×16`, tratate ca „tokeni", iar mecanismul de self-attention modelează relațiile dintre regiuni îndepărtate ale imaginii.
- **Rețele neuronale pe grafuri (GNN / Graph CNN):** imaginea este reprezentată ca un graf `G = (V, E, X)`, unde nodurile `V` sunt regiuni/patch-uri (sau zone semnificative medical), muchiile `E` codifică vecinătatea sau similaritatea, iar `X` conține trăsăturile fiecărui nod. Modelul propagă informație între noduri pentru a clasifica întregul graf sau fiecare nod.
- **Transfer learning:** modelele sunt pre-antrenate pe seturi mari de imagini (de ex. ImageNet) și apoi **fine-tunate** pe mamografii, ceea ce reduce necesarul de date medicale etichetate, care sunt greu de obținut.

---

## 4. Ce anume se rezolvă

Se rezolvă **detectarea și caracterizarea automată a tumorilor mamare** (benigne vs. maligne) în imagini de mamografie, cu scopul **diagnosticării timpurii** a cancerului de sân.

Dificultăți specifice problemei:

- variații mari ale densității țesutului mamar (sânii denși „maschează" tumorile);
- structuri suprapuse în proiecția 2D;
- diferențe subtile între leziunile benigne și cele maligne;
- leziuni foarte mici față de dimensiunea imaginii (de ordinul a câțiva pixeli dintr-o imagine de mii de pixeli);
- dezechilibru puternic al claselor și număr limitat de date etichetate.

## 5. De ce este important

- **Impact asupra vieții:** cancerul de sân este printre cele mai frecvente tipuri de cancer la femei, iar depistarea într-un stadiu incipient crește semnificativ șansele de vindecare și permite tratamente mai puțin agresive.
- **Limitele analizei manuale:** interpretarea de către radiologi este consumatoare de timp, obositoare și predispusă la erori umane (cazuri ratate sau alarme false), iar între medici apar variații de interpretare.
- **Volumul mare de screening:** programele de screening generează un număr mare de imagini, în timp ce numărul de radiologi este limitat.
- **Reducerea biopsiilor inutile:** o clasificare mai bună între benign și malign poate reduce procedurile invazive și stresul pacientelor.
- **Egalizarea accesului:** un sistem de sprijin decizional poate ajuta și unitățile medicale cu mai puțină experiență de specialitate.

## 6. Cine sunt utilizatorii

| Utilizator | Cum folosește sistemul |
|---|---|
| **Medici radiologi** (utilizatori principali) | Primesc o „a doua opinie" automată, zone marcate ca suspecte și un scor de risc |
| **Medici oncologi / senologi** | Folosesc rezultatul pentru prioritizarea cazurilor și decizia privind investigații suplimentare |
| **Unități medicale și programe de screening** | Triază rapid un volum mare de examinări |
| **Paciente** (beneficiari indirecți) | Beneficiază de un diagnostic mai rapid și mai precis |

Decizia finală rămâne întotdeauna la medic.

## 7. Date de intrare și de ieșire

### Date de intrare

- **Tip:** imagini medicale, adică mamografii digitale (tonuri de gri, rezoluție mare, de obicei în format DICOM sau PGM/PNG după conversie).
- **Variante:** imagini 2D (vizualizări CC și MLO) și, opțional, tomosinteză digitală (**DBT**, volum 3D de felii).
- **Metadate / adnotări (pentru antrenare):** etichete benign/malign, categorii BI-RADS, poziția și conturul leziunii (bounding box sau mască), densitatea sânului.
- **Preprocesare tipică:** decupare, normalizare a intensității, redimensionare, augmentare (rotiri, flip-uri, modificări de contrast), împărțirea în patch-uri sau construirea grafului.

**Seturi de date publice propuse:**

| Set de date | Descriere scurtă |
|---|---|
| **MIAS** | Set clasic, mic (322 de imagini), cu adnotări ale anomaliilor |
| **DDSM / CBIS-DDSM** | Set mare de mamografii de screening, cu leziuni adnotate și diagnostic confirmat |
| **INbreast** | Mamografii digitale de calitate înaltă (410 imagini), cu adnotări precise |
| **Breast Cancer Screening DBT** | Date de tomosinteză digitală a sânului (volume 3D) |

### Date de ieșire

- **Eticheta de clasă:** benign / malign (sau categoria BI-RADS);
- **Probabilitatea / scorul de risc** asociat predicției (`p ∈ [0, 1]`);
- **Localizarea leziunii:** chenar (bounding box) sau hartă de atenție / mască de segmentare, pentru a arăta medicului *unde* se află zona suspectă și pentru interpretabilitate.

## 8. Cum se măsoară performanța

Evaluarea se face pe un set de test separat (preferabil **la nivel de pacientă**, pentru a evita scurgerea de informație între antrenare și test), eventual cu validare încrucișată.

**Metrici de clasificare**

| Metrică                                     | Ce spune                                              | De ce contează aici                                                        |
| ------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| **Sensibilitate (Recall)** `TP / (TP + FN)` | Câte cazuri maligne sunt detectate                    | Cea mai importantă: un fals negativ înseamnă un cancer ratat               |
| **Specificitate** `TN / (TN + FP)`          | Câte cazuri sănătoase/benigne sunt recunoscute corect | Reduce alarmele false și biopsiile inutile                                 |
| **Precizie (Precision)** `TP / (TP + FP)`   | Din cazurile semnalate, câte sunt într-adevăr maligne | Încrederea în alerte                                                       |
| **F1-score**                                | Media armonică a preciziei și recall-ului             | Util la clase dezechilibrate                                               |
| **Acuratețe**                               | Procentul total de predicții corecte                  | Informativă doar împreună cu celelalte, din cauza dezechilibrului claselor |
| **AUC-ROC / AUC-PR**                        | Calitatea separării claselor, independent de prag     | Metrica standard în literatura medicală                                    |
| **Matrice de confuzie**                     | Distribuția completă a erorilor                       | Analiza tipurilor de erori                                                 |

**Metrici de localizare / segmentare (dacă este cazul)**

- **IoU** (Intersection over Union) și **Dice coefficient**, pentru suprapunerea dintre zona prezisă și cea adnotată de medic;
- **FROC** (Free-Response ROC), adică sensibilitatea în funcție de numărul de false pozitive per imagine.

**Criterii suplimentare**

- **Generalizare:** performanța la antrenarea pe un set de date și testarea pe altul (de ex. CBIS-DDSM → INbreast);
- **Calibrare:** măsura în care probabilitățile prezise corespund frecvenței reale (de ex. Expected Calibration Error);
- **Interpretabilitate:** calitatea hărților de atenție (Grad-CAM, attention maps) evaluată împreună cu medicul;
- **Comparația cu radiologii:** performanța modelului față de cea a medicilor, singur și ca sistem de asistare;
- **Eficiență:** timpul de inferență și resursele necesare;
- **Comparația cu modele de referință (baseline):** CNN clasice, ViT, DeiT, CrossViT, MammoViT și variantele bazate pe GNN.

---
