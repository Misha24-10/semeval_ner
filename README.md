# SEMEVAL NER

Данный репозиторий содержит реализацию моделей для многоязычного распознавания именованных сущностей на основе RemBERT и XLM-RoBERTa. Также предоставлены скрипты для обучения, тестирования и ансамблирования моделей. Репозиторий может быть полезен в области обработки естественного языка, машинного перевода, анализа текстов и автоматического извлечения информации.

Структура данного репозитория выглядит следующим образом:

## Описание файлов

- `configuration.py`: конфигурационный файл, определяющий параметры для запуска моделей.
- `eval_ensemble_models.py`: запускает модуль тестирования ансамбля моделей на валидационной/тестовой выборке.
- `eval__nn_based_ensemble_models.py`: запускает модуль тестирования ансамбля моделей, основанного на использовании мета-модели, на валидационной/тестовой выборке.
- `train_eval_nnensemble_models.py`: запускает модуль обучения мета-модели.
- `train_ner_template.py`: запускает модуль для обучения модели на основе RemBERT.
- `train_ner_template_xlm_roberta.py`: запускает модуль для обучения модели на основе XLM-RoBERTa.

### Пакеты

- `metric`: содержит метрики для вычисления промежуточного F1-счета во время обучения моделей.
- `models`: содержит классы моделей.
- `reader`: используется для предобработки данных.

## Результаты
[WanDB project page](https://wandb.ai/mishaya/NER%20multilangual?workspace=user-mishaya "WanDB project page")

Дообученные модели:

–	Model_1 = [google–rembert–ft_for_multi_ner_v3](https://drive.google.com/file/d/1IRrip01noCquGDDFcV14Qw-SRjUwkXHB/view?usp=share_link "google–rembert–ft_for_multi_ner_v3");

–	Model_2 = [xlm_roberta_large_mountain](https://drive.google.com/file/d/1EPObulw4HYZngsM3Bf0IfHNBruTz42Tz/view?usp=share_link);

–	Model_3 = [google–rembert–ft_for_multi_ner_sky](https://drive.google.com/file/d/18-GOgvwAjC39_HajBnalgshmy14jd-MR/view?usp=share_link).

Ансамбль этих моделей достигает качества F1_score=0.73 на тестовом сете датасета MultiCoNER II

