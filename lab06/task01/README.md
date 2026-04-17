# Template projektu ML: klasyfikacja Iris w PyTorch

To repozytorium ma ci pokazać, jak zorganizować mały projekt Machine Learning tak, żeby był czytelny, rozszerzalny i zbliżony do pracy nad realnym projektem, a nie tylko do jednorazowego notebooka.

Domena jest celowo prosta: klasyfikacja Iris. Dzięki temu możesz skupić się nie na samym zadaniu ML, ale na strukturze projektu, konfiguracji, warstwie eksperymentów, zapisie artefaktów i porównywaniu wyników.

W aktualnej wersji template'u masz:

- dwa klasyfikatory,
- dwie strategie preprocessingu,
- cztery gotowe konfiguracje eksperymentów,
- osobny krok porównywania wyników.

## Jak uruchomić/korzystać z template?

### 1. Co pokazuje ten template?

Projekt demonstruje pełny, uporządkowany przepływ pracy:

1. pobranie danych i zapis do `data/raw/`,
2. preprocessing, czyli przygotowanie danych do modelu,
3. trening modelu,
4. zapis checkpointu i artefaktów,
5. ewaluację,
6. wizualizację,
7. pojedynczą predykcję,
8. uruchamianie wielu eksperymentów `src/experiments`,
9. porównywanie wyników eksperymentów,
10. testy, linting, type-checking i hooki `pre-commit`.

Ten pipeline jest bardzo przejrzysty i wygody w użyciu, zwłaszcza gdy projekt ML przestaje być "jednym modelem" i  zaczynasz porównywać różne warianty preprocessingu, różne klasyfikatory i różne ustawienia treningu.  

### 2. Jak wygląda struktura katalogów?

Najważniejsze katalogi:

- `configs/` – konfiguracje eksperymentów w YAML.
- `data/raw/` – dane surowe.
- `data/processed/` – dane po preprocessingu, zapisywane w podkatalogach per eksperyment.
- `models/` – checkpointy modeli i artefakty preprocessingu, osobno dla każdego eksperymentu.
- `reports/` – raporty, predykcje i wykresy per eksperyment oraz wspólne raporty globalne.
- `src/cli/` – punkty wejścia uruchamiane z terminala.
- `src/data/` – ładowanie danych i preprocessing.
- `src/models/` – definicje modeli.
- `src/experiments/` – warstwa eksperymentów.
- `src/training/` – wspólna logika treningu i predykcji.
- `src/visualization/` – generowanie wykresów.
- `src/utils/` – konfiguracja, logowanie, I/O (ang. input/output, czyli odczyt i zapis danych), seed, przygotowanie runtime.
- `tests/` – testy automatyczne.

### 3. Jak przygotować środowisko?

Projekt zakłada użycie `uv`, czyli narzędzia do zarządzania środowiskiem i zależnościami (ang. dependencies).

Przykładowo:

```bash
uv sync --group dev
```

albo:

```bash
make setup
```

Po instalacji zależności możesz uruchamiać komendy przez `uv run`.

### 4. Czym są eksperymenty?

W tym projekcie eksperyment to konkretny wariant pipeline'u ML: określony model, określona strategia preprocessingu, zestaw parametrów treningu oraz ścieżki zapisu artefaktów i raportów.

Elementy eksperymentu są rozdzielone między kilka warstw projektu:

- `configs/` – konfiguracja eksperymentu w YAML,
- `src/models/` – definicje modeli,
- `src/data/` – logika ładowania danych i preprocessingu,
- `src/training/` – wspólna logika treningu, ewaluacji i predykcji,
- `src/experiments/` – warstwa spinająca te elementy w jeden uruchamialny eksperyment.

To znaczy, że `src/experiments/` nie przechowuje wszystkich "klocków" osobno, tylko wybiera i łączy komponenty z innych części projektu.

Dzięki temu:

- konfiguracja jest oddzielona od logiki,
- modele i preprocessing nie są kopiowane między eksperymentami,
- różne warianty można łatwo uruchamiać i porównywać.

Każdy eksperyment korzysta z tego samego ogólnego interfejsu: preprocessing, trening, ewaluacja i predykcja, ale konkretne ustawienia wynikają z pliku YAML i typu eksperymentu.

### 4a. Gdzie w kodzie łączą się model i preprocessing?

To jest ważne, bo na pierwszy rzut oka może się wydawać, że skoro istnieją konfiguracje `mlp + zscore` i `mlp + minmax`, to w `src/experiments/` powinny istnieć osobne klasy dla każdej takiej kombinacji.

W tym template tak nie jest.

Połączenie modelu i preprocessingu powstaje dynamicznie z dwóch źródeł:

- klasa eksperymentu wybiera model,
- plik YAML wybiera strategię preprocessingu.

Przykładowo:

- `model.name: mlp_classifier` powoduje wybór klasy eksperymentu MLP,
- `preprocessing.strategy: zscore` albo `minmax` wybiera sposób przygotowania danych.

Te dwie rzeczy spotykają się w klasie bazowej eksperymentu:

- `build_model()` dostarcza konkretny model,
- `preprocess()` używa strategii z konfiguracji,
- `run()` uruchamia cały pipeline we właściwej kolejności.

To znaczy, że:

- `src/experiments/` odpowiada głównie za wybór wariantu modelowego,
- `src/data/` odpowiada za preprocessing,
- `src/experiments/base.py` skleja te elementy w jeden przebieg eksperymentu.

Takie podejście ogranicza duplikację kodu i dobrze działa wtedy, gdy chcesz niezależnie łączyć modele z różnymi strategiami preprocessingu.

Istnieje jednak także inne sensowne podejście: można potraktować cały wariant `model + preprocessing` jako osobny typ eksperymentu.

Wtedy mogłyby istnieć klasy takie jak:

- `MLPZScoreExperiment`,
- `MLPMinMaxExperiment`,
- `LinearZScoreExperiment`.

Takie rozwiązanie ma sens wtedy, gdy:

- nie chcesz wspierać wszystkich możliwych kombinacji, tylko kilka wybranych,
- każda kombinacja ma własną logikę lub własne założenia,
- preprocessing jest traktowany jako część scenariusza eksperymentalnego, a nie tylko parametr techniczny.

W takim wariancie warstwa `src/experiments/` opisywałaby gotowe scenariusze badawcze, a nie tylko wybór modelu.

Ten template pokazuje jednak inne podejście:

- model jest wybierany przez klasę eksperymentu,
- preprocessing jest wybierany przez konfigurację,
- wspólny pipeline pozostaje jeden.

Dzięki temu łatwiej pokazać studentowi rozdział odpowiedzialności między kodem a konfiguracją i uniknąć mnożenia bardzo podobnych klas.

### 4b. Diagram przepływu eksperymentu

Poniższy uproszczony schemat pokazuje, jak konfiguracja przechodzi przez kod:

```text
configs/iris_mlp_zscore.yaml
    |
    |  wczytanie konfiguracji
    v
ProjectConfig
    |
    |  model.name = "mlp_classifier"
    |  preprocessing.strategy = "zscore"
    v
ExperimentFactory.build(config.model.name, config)
    |
    |  wybór klasy eksperymentu
    v
MLPClassifierExperiment
    |
    |  dziedziczy wspólny workflow z BaseExperiment
    v
BaseExperiment.run(frame)
    |
    +--> preprocess()
    |       |
    |       +--> src/data/preprocessing.py
    |             używa config.preprocessing.strategy
    |
    +--> train()
    |       |
    |       +--> build_model()
    |             zwraca model z src/models/
    |
    +--> evaluate()
    |
    +--> visualize_training()
```

W skrócie:

- YAML mówi, jaki model i jaki preprocessing wybrać,
- fabryka eksperymentów wybiera odpowiednią klasę eksperymentu,
- klasa bazowa wykonuje wspólny pipeline,
- konkretna klasa eksperymentu dostarcza model,
- preprocessing jest brany z konfiguracji.

### 4c. Inne sposoby modelowania eksperymentów

Warto pamiętać, że to nie jest jedyny sensowny sposób organizacji eksperymentów.

W tym template eksperyment oznacza pełny przebieg end-to-end:

- preprocessing,
- trening,
- ewaluację,
- zapis artefaktów i raportów.

W tym wariancie łatwo prześledzić cały przepływ od danych wejściowych do wyniku końcowego.

W innych projektach można jednak przyjąć inną definicję eksperymentu.

Jedna możliwość to eksperymenty etapowe, na przykład:

- eksperyment preprocessingu,
- eksperyment treningu,
- eksperyment ewaluacji.

Takie podejście ma sens, gdy:

- poszczególne etapy są kosztowne i chcesz uruchamiać je niezależnie,
- jeden preprocessing ma być używany przez wiele treningów,
- chcesz porównywać same strategie przygotowania danych,
- pipeline jest bardziej rozbudowany niż w tym przykładzie.

Można też traktować eksperyment jako pojedynczą decyzję badawczą, na przykład:

- porównanie dwóch strategii preprocessingu,
- porównanie dwóch modeli,
- porównanie dwóch schedulerów lub optymalizatorów.

W takim podejściu eksperyment nie musi oznaczać pełnego pipeline'u, tylko kontrolowaną zmianę jednego elementu względem ustalonej bazy.

Jeszcze inna opcja to modelowanie eksperymentu jako gotowego scenariusza end-to-end, w którym klasa eksperymentu definiuje jednocześnie model, preprocessing i inne reguły wykonania.

Ten template pokazuje najprostszy wariant do nauki:

- eksperyment = pełny przebieg,
- model wybierany przez klasę eksperymentu,
- preprocessing wybieray przez konfiguracja.

To nie jest jedyne poprawne rozwiązanie, ale jest dobrym punktem startowym do nauki uporządkowanej struktury projektu ML.

### 5. Jakie eksperymenty są już przygotowane?

Masz cztery gotowe konfiguracje eksperymentów:

- [configs/iris_mlp_zscore.yaml](configs/iris_mlp_zscore.yaml) – model `mlp_classifier`, preprocessing `zscore`
- [configs/iris_mlp_minmax.yaml](configs/iris_mlp_minmax.yaml) – model `mlp_classifier`, preprocessing `minmax`
- [configs/iris_linear_zscore.yaml](configs/iris_linear_zscore.yaml) – model `linear_classifier`, preprocessing `zscore`
- [configs/iris_linear_minmax.yaml](configs/iris_linear_minmax.yaml) – model `linear_classifier`, preprocessing `minmax`

Czyli porównujesz:

- dwa klasyfikatory:
  - MLP (ang. Multi-Layer Perceptron),
  - model liniowy,
- dwie strategie preprocessingu:
  - `zscore`,
  - `minmax`.

Każdy z tych eksperymentów używa tej samej wspólnej logiki pipeline'u, ale innej konfiguracji i innego zestawu komponentów.

To jest minimalny, ale bardzo sensowny przykład eksperymentowania.


### 6. Jak uruchomić pełny zestaw eksperymentów?

Najprostsza ścieżka:

```bash
make pipeline
```

Ta komenda:

1. pobierze dane,
2. uruchomi cztery eksperymenty,
3. wygeneruje wykresy danych surowych,
4. stworzy raport porównawczy.

### 7. Jak uruchamiać eksperymenty pojedynczo?

Najpierw pobierz dane:

```bash
make fetch
```

albo:

```bash
uv run -m src.cli.fetch_data --config configs/iris_mlp_zscore.yaml
```

Potem możesz odpalić pojedynczy eksperyment:

```bash
uv run -m src.cli.run_experiment --config configs/iris_mlp_zscore.yaml
uv run -m src.cli.run_experiment --config configs/iris_mlp_minmax.yaml
uv run -m src.cli.run_experiment --config configs/iris_linear_zscore.yaml
uv run -m src.cli.run_experiment --config configs/iris_linear_minmax.yaml
```

Masz też skróty w `Makefile`:

- `make run-experiment`
- `make run-mlp-zscore`
- `make run-mlp-minmax`
- `make run-linear-zscore`
- `make run-linear-minmax`

### 8. Jak porównywać eksperymenty?

Do porównania służy:

```bash
make compare
```

albo:

```bash
uv run -m src.cli.compare_experiments --config configs/compare_experiments.yaml
```

Wyniki trafiają do:

- [experiment_comparison.csv](reports/comparisons/experiment_comparison.csv)
- [experiment_comparison.json](reports/comparisons/experiment_comparison.json)
- [experiment_comparison.png](reports/comparisons/experiment_comparison.png)

To jest ważna część template'u. Projekt ML nie kończy się na "udało się wytrenować model". Musisz jeszcze umieć zebrać wyniki i porównać warianty w sposób czytelny dla siebie i innych.

### 9. Co dokładnie porównujesz?

Każdy eksperyment zapisuje między innymi:

- `training_summary.json`
- `evaluation.json`
- `training_history.csv`
- `test_predictions.csv`

Pliki te trafiają do katalogu raportowego danego eksperymentu, na przykład `reports/iris_mlp_zscore/`.

Warstwa porównawcza zbiera z tych plików najważniejsze metryki, na przykład:

- najlepszą accuracy na walidacji,
- accuracy na zbiorze testowym,
- nazwę klasyfikatora,
- nazwę strategii preprocessingu.

To daje ci prosty, ale bardzo praktyczny wzorzec:

- eksperymenty zapisują wyniki lokalnie,
- osobny krok agreguje wyniki,
- raport porównawczy powstaje automatycznie.

### 10. Co to jest CLI i dlaczego tu występuje?

CLI (ang. Command Line Interface) to interfejs uruchamiany z terminala, czyli zestaw komend wywoływanych z linii poleceń.

W tym template CLI jest wydzielone do `src/cli/`, bo taki układ:

- jasno rozdziela etapy pracy,
- daje ci wygodne punkty wejścia,
- dobrze skaluje się do większych projektów,
- ułatwia automatyzację w CI (ang. Continuous Integration, czyli automatycznym sprawdzaniu projektu po zmianach).

Zamiast uruchamiać ręcznie funkcje z plików albo notebooków, masz stabilny interfejs do obsługi projektu.

### 11. Skąd biorą się dane?

W tym szablonie dane `Iris Dataset` pochodzą z `scikit-learn`, dokładniej z `sklearn.datasets.load_iris`, bo jest to stabile i sprawdzone źródło. Są one nadal są zapisywane lokalnie do `data/raw/iris.csv`, więc zachowujesz klasyczny układ warstw danych. 

### 12. Co dokładnie robi preprocessing?

W tym template preprocessing:

1. dzieli dane na `train`, `validation`, `test`,
2. uczy parametry transformacji tylko na zbiorze treningowym,
3. stosuje tę samą transformację do pozostałych splitów,
4. zapisuje dane przetworzone do osobnego katalogu eksperymentu,
5. zapisuje artefakt preprocessingu.

Masz dwie strategie:

- `zscore` – centrowanie i skalowanie względem średniej i odchylenia standardowego,
- `minmax` – przeskalowanie cech do zakresu od 0 do 1.

To dobry przykład, bo możesz zobaczyć, że preprocessing też jest częścią eksperymentu, a nie tylko "technicznym szczegółem".

### 13. Jak działa trening?

Każdy eksperyment:

1. wybiera model,
2. czyta swoją konfigurację,
3. ładuje własne dane przetworzone,
4. trenuje model,
5. zapisuje checkpoint i historię treningu,
6. zapisuje podsumowanie do JSON.

To znaczy, że:

- logika treningu jest wspólna,
- wybór modelu i strategii preprocessingu zależy od eksperymentu,
- wyniki są od siebie oddzielone katalogami.

To bardzo dobry wzorzec dla początkującej osoby, bo pokazuje ci, jak współdzielić kod bez mieszania wyników.

### 14. Jakie modele są tu użyte?

Masz dwa klasyfikatory:

`mlp_classifier`

- prosty model MLP,
- ma warstwy ukryte,
- jest nieliniowy,
- dobrze pokazuje typowy trening małej sieci neuronowej.

`linear_classifier`

- bardzo prosty klasyfikator liniowy,
- nie ma ukrytych warstw,
- jest dobrym punktem odniesienia,
- pozwala ci zobaczyć, czy bardziej złożony model naprawdę daje zysk.

To jest bardzo ważne w projektach ML: warto porównywać model "bardziej ambitny" z prostszą bazą.

### 15. Jak działa ewaluacja i predykcja?

Każdy eksperyment:

- ładuje własny checkpoint,
- ładuje własne dane testowe,
- liczy metryki,
- zapisuje predykcje,
- generuje macierz pomyłek.

Pojedyncza predykcja działa przez:

```bash
make predict-example
```

albo:

```bash
uv run -m src.cli.predict --config configs/iris_mlp_zscore.yaml --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

Predykcja korzysta z artefaktu preprocessingu danego eksperymentu. To ważne, bo w dobrze zbudowanym projekcie inferencja powinna używać dokładnie tej samej transformacji, której użyto przy treningu.

### 16. Jak dbać o jakość kodu?

Dostępne komendy:

```bash
make test
make ruff
make format
make pyright
```

`pytest`

- służy do testów automatycznych,
- pozwala ci sprawdzić, czy zmiany nie popsuły istniejącej logiki.

`ruff`

- służy do lintingu, czyli automatycznego wykrywania problemów stylistycznych, niespójności i części błędów,
- potrafi też formatować kod.

`pyright`

- służy do type-checkingu (statycznej analizy typów),
- sprawdza, czy dane przekazywane między funkcjami zgadzają się z deklarowanymi typami.

`pre-commit`

- uruchamia kontrole jakości automatycznie przed commitem i pushem,
- nie musisz pamiętać o ręcznym odpalaniu każdego narzędzia.

To są narzędzia, które dobrze pokazać w template'cie, bo uczą nie tylko pisania kodu, ale też pracy nad projektem w sposób uporządkowany.

### 17. Troubleshooting dla Windows

Jeśli pracujesz na Windowsie, możesz trafić na dwa problemy środowiskowe.

`make` nie jest rozpoznawane

- Windows zwykle nie ma `make` zainstalowanego domyślnie,
- jeśli `make setup`, `make pipeline` albo `make compare` nie działa, zainstaluj `make` dla Windows z tej instrukcji:
- https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make

Jeśli nie chcesz instalować `make`, nadal możesz wykonywać wszystkie kroki przez `uv run`.

PyTorch zgłasza błąd DLL

- na Windowsie biblioteki binarne czasem nie ładują się poprawnie mimo poprawnej instalacji,
- w tym template został dodany kod pomocniczy przygotowujący środowisko przed importem `torch`.

Jeśli problem nadal występuje:

1. uruchom ponownie terminal,
2. wykonaj ponownie `uv sync --group dev`,
3. upewnij się, że korzystasz z właściwego interpretera i `.venv`.

## Przegląd technologii użytych w template

Poniższy przegląd ma ci pokazać nie tylko listę bibliotek, ale też rolę, jaką każda z nich pełni w uporządkowanym projekcie ML. W małym projekcie wiele narzędzi można pominąć, ale wtedy trudniej budować kod, który da się rozwijać, testować i przekazać dalej.

### Python

Link: https://www.python.org/

Po co jest użyty:

- to główny język całego projektu,
- w Pythonie są napisane skrypty CLI, warstwa konfiguracji, preprocessing, modele, trening, ewaluacja i wizualizacje,
- dobrze nadaje się do łączenia logiki ML z narzędziami inżynierskimi.

Dlaczego ma sens:

- możesz zbudować cały pipeline w jednym ekosystemie,
- nie rozdzielasz projektu na osobny kod badawczy i osobny kod "narzędziowy",
- to typowy wybór w projektach machine learning i data science.

Co to daje ci w praktyce:

- łatwiej utrzymać spójność projektu,
- łatwiej czytać kod, gdy wszystkie etapy są zapisane w jednym języku.

### uv

Link: https://docs.astral.sh/uv/

Po co jest użyty:

- do zarządzania środowiskiem wirtualnym,
- do instalacji zależności (ang. dependencies),
- do uruchamiania komend projektu przez `uv run`.

Dlaczego ma sens:

- daje ci powtarzalne środowisko,
- ogranicza problemy typu "u mnie działa, a u kogoś innego nie działa",
- upraszcza pracę bardziej niż ręczne pilnowanie `venv` i `pip`.

Co to daje ci w praktyce:

- możesz łatwiej odtworzyć projekt na nowym komputerze,
- masz jeden spójny sposób instalacji i uruchamiania wszystkiego.

### PyTorch

Link: https://pytorch.org/

Po co jest użyty:

- do definicji modeli `mlp_classifier` i `linear_classifier`,
- do treningu modeli,
- do zapisu wag modelu,
- do inferencji (ang. inference), czyli wykonywania predykcji na nowych danych.

Dlaczego ma sens:

- pozwala ci zobaczyć pełny cykl pracy z modelem,
- umożliwia zapis artefaktu modelu i późniejsze ponowne użycie tego samego klasyfikatora,
- jest szeroko używany w praktyce, więc poznajesz narzędzie realnie obecne w projektach produkcyjnych i badawczych.

Co to daje ci w praktyce:

- uczysz się, że model nie kończy się na samym treningu,
- widzisz, jak połączyć trening, zapis checkpointu i predykcję w jeden workflow.

### scikit-learn

Link: https://scikit-learn.org/

Po co jest użyty:

- jako źródło danych Iris,
- do podziału danych na `train`, `validation` i `test`,
- do metryk ewaluacyjnych, na przykład accuracy i confusion matrix.

Dlaczego ma sens:

- dane są stabilne i łatwo dostępne,
- nie musisz organizować osobnego źródła danych tylko po to, żeby ćwiczyć strukturę projektu,
- biblioteka dostarcza też sprawdzone narzędzia pomocnicze do ewaluacji.

Co to daje ci w praktyce:

- możesz skupić się na architekturze projektu,
- od razu pracujesz na pipeline'ie, a nie na problemach z pobieraniem danych.

### pandas

Link: https://pandas.pydata.org/

Po co jest użyty:

- do pracy na danych tabelarycznych,
- do zapisu surowych i przetworzonych danych do CSV,
- do zapisu predykcji i raportów,
- do agregacji wyników eksperymentów w jeden plik porównawczy.

Dlaczego ma sens:

- w projektach ML bardzo często operujesz na tabelach, metrykach i raportach,
- `pandas` daje prosty interfejs do pracy z takimi strukturami.

Co to daje ci w praktyce:

- uczysz się rozdzielać dane wejściowe, dane po preprocessingu i wyniki,
- widzisz, że CSV nadal jest bardzo użytecznym formatem roboczym.

### Pydantic

Link: https://docs.pydantic.dev/

Po co jest użyty:

- do definiowania struktury konfiguracji,
- do walidacji plików YAML,
- do sprawdzania, czy kod dostaje komplet poprawnych parametrów.

Dlaczego ma sens:

- bez walidacji konfiguracji łatwo o błędy typu zła nazwa pola, brak wartości albo zły typ,
- takie błędy lepiej wykryć przed startem treningu niż w połowie pipeline'u.

Co to daje ci w praktyce:

- możesz zmieniać parametry bez grzebania w kodzie,
- szybciej rozumiesz, jakie pola są wymagane,
- łatwiej rozbudujesz projekt o kolejne eksperymenty.

### PyYAML

Link: https://pyyaml.org/

Po co jest użyty:

- do wczytywania plików YAML z konfiguracją eksperymentów,
- do ładowania konfiguracji porównania eksperymentów.

Dlaczego ma sens:

- YAML jest czytelny i wygodny w edycji,
- dobrze sprawdza się jako warstwa parametrów trzymanych poza kodem.

Co to daje ci w praktyce:

- możesz uruchamiać różne warianty eksperymentu bez przepisywania logiki,
- łatwiej widzisz różnice między konfiguracjami.

### argparse

Link: https://docs.python.org/3/library/argparse.html

Po co jest użyty:

- do budowy CLI (ang. Command Line Interface), czyli interfejsu uruchamianego z terminala,
- do przekazywania argumentów, takich jak ścieżka do konfiguracji albo pojedyncze cechy wejściowe do predykcji.

Dlaczego ma sens:

- zamiast wywoływać ręcznie funkcje w kodzie, masz stabilne punkty wejścia do projektu,
- taki układ lepiej skaluje się do większych projektów niż uruchamianie wszystkiego z notebooka.

Co to daje ci w praktyce:

- łatwiej automatyzujesz powtarzalne kroki,
- łatwiej przekazujesz projekt innej osobie,
- łatwiej podpinasz pipeline pod CI lub skrypty pomocnicze.

### pytest

Link: https://docs.pytest.org/

Po co jest użyty:

- do testów automatycznych,
- do sprawdzania, czy konfiguracja, preprocessing i warstwa eksperymentów działają zgodnie z oczekiwaniem.

Krótka definicja:

- test automatyczny to mały fragment kodu, który sprawdza konkretne zachowanie programu bez ręcznego klikania lub ręcznego uruchamiania całego pipeline'u.

Dlaczego ma sens:

- gdy coś zmienisz, możesz szybko sprawdzić, czy nie popsułeś istniejącej logiki,
- to szczególnie ważne, kiedy projekt zaczyna mieć kilka etapów i kilka konfiguracji.

Co to daje ci w praktyce:

- większe bezpieczeństwo zmian,
- mniej ręcznego sprawdzania po każdej edycji kodu.

### Ruff

Link: https://docs.astral.sh/ruff/

Po co jest użyty:

- do lintingu, czyli automatycznego wykrywania problemów stylistycznych, niespójności i części prostych błędów,
- do formatowania kodu.

Dlaczego ma sens:

- początkujące projekty szybko stają się niespójne, jeśli każdy plik wygląda inaczej,
- linter i formatter przejmują część mechanicznej kontroli jakości.

Co to daje ci w praktyce:

- mniej czasu tracisz na ręczne poprawianie stylu,
- szybciej zauważasz część problemów jeszcze przed uruchomieniem programu.

### Pyright

Link: https://microsoft.github.io/pyright/

Po co jest użyty:

- do type-checkingu (statycznej analizy typów),
- do sprawdzania, czy dane przekazywane między funkcjami zgadzają się z deklarowanymi typami.

Krótka definicja:

- statyczna analiza typów to sprawdzanie kodu bez jego uruchamiania pod kątem zgodności argumentów, zwracanych wartości i struktur danych.

Dlaczego ma sens:

- pozwala wychwycić część błędów wcześniej,
- poprawia czytelność interfejsów między modułami,
- pomaga pisać bardziej precyzyjny kod.

Co to daje ci w praktyce:

- łatwiej rozumiesz, co dana funkcja przyjmuje i zwraca,
- mniejsza szansa, że przez nieuwagę pomylisz format danych między etapami pipeline'u.

### pre-commit

Link: https://pre-commit.com/

Po co jest użyty:

- do automatycznego uruchamiania kontroli jakości przed commitem i przed pushem,
- do spinania narzędzi takich jak Ruff, Pyright i testy w jeden mechanizm.

Krótka definicja:

- hook to automatyczna akcja wykonywana przy określonym zdarzeniu, na przykład przed wykonaniem commita.

Dlaczego ma sens:

- nie musisz pamiętać o ręcznym uruchamianiu wszystkich kontroli,
- projekt sam pilnuje minimalnego poziomu jakości.

Co to daje ci w praktyce:

- wyrabiasz sobie dobry nawyk pracy z kodem,
- szybciej zauważasz problemy, zanim trafią do repozytorium.

### Makefile

Link: https://www.gnu.org/software/make/manual/make.html

Po co jest użyty:

- jako warstwa skrótów do najważniejszych zadań projektu,
- do uruchamiania powtarzalnych kroków przez krótkie komendy, na przykład `make train`, `make pipeline` albo `make compare`.

Dlaczego ma sens:

- ukrywa dłuższe komendy za krótkimi nazwami,
- ułatwia wejście do projektu nowej osobie,
- porządkuje sposób pracy całego zespołu.

Co to daje ci w praktyce:

- nie musisz pamiętać pełnych komend,
- łatwiej pokazujesz komuś innemu, jak uruchamia się projekt.

### YAML jako warstwa konfiguracji

Link: https://yaml.org/

Po co jest użyty:

- do opisywania parametrów eksperymentu poza kodem,
- do przechowywania informacji o modelu, preprocessingu, ścieżkach i hiperparametrach.

Dlaczego ma sens:

- oddzielenie konfiguracji od logiki to jeden z podstawowych wzorców dobrze zorganizowanego projektu,
- dzięki temu łatwiej porównywać eksperymenty i śledzić różnice między uruchomieniami.

Co to daje ci w praktyce:

- możesz mieć kilka wariantów eksperymentów bez kopiowania kodu,
- łatwiej utrzymujesz porządek w parametrach.

### JSON i CSV jako artefakty

Link do JSON: https://www.json.org/json-en.html

Link do CSV: https://datatracker.ietf.org/doc/html/rfc4180

Po co są użyte:

- CSV do danych, predykcji, historii treningu i raportów tabelarycznych,
- JSON do metadanych, podsumowań i wyników zapisanych w formie łatwej do odczytu przez kod.

Krótka definicja:

- artefakt to plik będący wynikiem działania pipeline'u, na przykład model, raport, zapis metryk albo parametry preprocessingu.

Dlaczego ma sens:

- dobrze zbudowany projekt nie kończy się na `print()` w terminalu,
- wyniki powinny być zapisane tak, żeby można było do nich wrócić później.

Co to daje ci w praktyce:

- możesz analizować wyniki po czasie,
- możesz porównywać eksperymenty bez ponownego uruchamiania wszystkiego.

### Warstwa `src/experiments`

Link: to nie jest osobna biblioteka, tylko element architektury tego template'u.

Po co jest użyta:

- do wydzielenia wspólnego interfejsu eksperymentów,
- do rejestrowania różnych typów eksperymentów,
- do uruchamiania kilku wariantów modeli i preprocessingu w uporządkowany sposób.

Dlaczego ma sens:

- przy jednym modelu i jednej konfiguracji taka warstwa może być zbędna,
- ale gdy pojawia się kilka modeli i kilka strategii preprocessingu, warto oddzielić logikę eksperymentu od jego parametrów.

Co to daje ci w praktyce:

- widzisz uproszczoną wersję wzorca używanego w większych projektach ML,
- łatwiej dodasz kolejny klasyfikator albo nową strategię preprocessingu,
- porównywanie eksperymentów staje się naturalną częścią projektu, a nie zbiorem osobnych skryptów.

## Dlaczego ten template ma wartość produkcyjną?

Bo pokazuje ci nie tylko "jak uruchomić model", ale jak zorganizować projekt tak, żeby:

- mieć wiele eksperymentów,
- rozdzielać konfigurację od logiki,
- zapisywać artefakty,
- automatycznie porównywać wyniki,
- utrzymywać jakość kodu,
- rozwijać projekt bez przepisywania wszystkiego od zera.

To właśnie jest główny cel tego template'u.
