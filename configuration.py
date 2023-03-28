CONLLIOBV2 = {
    "B-Disease":0,
    "I-Disease":1,
    "B-Symptom":2,
    "I-Symptom":3,
    "B-AnatomicalStructure":4,
    "I-AnatomicalStructure":5,
    "B-MedicalProcedure":6,
    "I-MedicalProcedure":7,
    "B-Medication/Vaccine":8,
    "I-Medication/Vaccine":9,

    "B-OtherPROD":10,
    "I-OtherPROD":11,
    "B-Drink":12,
    "I-Drink":13,
    "B-Food":14,
    "I-Food":15,
    "B-Vehicle":16,
    "I-Vehicle":17,
    "B-Clothing":18,
    "I-Clothing":19,

    "B-OtherPER":20,
    "I-OtherPER":21,
    "B-SportsManager":22,
    "I-SportsManager":23,
    "B-Cleric":24,
    "I-Cleric":25,
    "B-Politician":26,
    "I-Politician":27,
    "B-Athlete":28,
    "I-Athlete":29,
    "B-Artist":30,
    "I-Artist":31,
    "B-Scientist":32,
    "I-Scientist":33,
    
    "B-ORG":34,
    "I-ORG":35,
    "B-TechCorp":36,
    "I-TechCorp":37,
    "B-CarManufacturer":38,
    "I-CarManufacturer":39,
    "B-SportsGRP":40,
    "I-SportsGRP":41,
    "B-AerospaceManufacturer":42,
    "I-AerospaceManufacturer":43,
    "B-OtherCorp":44,
    "I-OtherCorp":45,
    "B-PrivateCorp":46,
    "I-PrivateCorp":47,
    "B-PublicCorp":48,
    "I-PublicCorp":49,
    "B-MusicalGRP":50,
    "I-MusicalGRP":51,

    "B-OtherCW":52,
    "I-OtherCW":53,
    "B-Software":54,
    "I-Software":55,
    "B-ArtWork":56,
    "I-ArtWork":57,
    "B-WrittenWork":58,
    "I-WrittenWork":59,
    "B-MusicalWork":60,
    "I-MusicalWork":61,
    "B-VisualWork":62,
    "I-VisualWork":63,

    "B-Station":64,
    "I-Station":65,
    "B-HumanSettlement":66,
    "I-HumanSettlement":67,
    "B-OtherLOC":68,
    "I-OtherLOC":69,
    "B-Facility":70,
    "I-Facility":71,

    'O': 72
}


# Rembert config

# MODEL_NAME = 'google/rembert'
# config = dict(
#     model_name = MODEL_NAME,
#     LEARINGIN_RATE = 3e-5,
#     EPOCHS = 5,
#     BATCH_SIZE = 32,
#     TRAIN_VAL_SPLIT = 0.8,
#     num_warmup_steps = 3000,
#     CLIP_GRAD_VALUE = 5,
#     USE_CLIP_GRAD = True,
#     optimizer = "AdamW",
#     max_length = 196,
#     num_cycles = 2,
#     sheculer = 'get_cosine_with_hard_restarts_schedule_with_warmup'
# )


# files_configs = dict(
#     train_path = "./public_data/MULTI_Multilingual/multi_train.conll",
#     test_path = "./public_data/MULTI_Multilingual/multi_dev.conll",

#     wandb_run_name = "google/rembert_v3",
#     wandb_notes = "rembert",
#     base_model_path = "./rembert",
#     res_path = "google-rembert-ft_for_multi_ner_v3"
# )


# xlm-roberta-large config

MODEL_NAME = 'xlm-roberta-large'
config = dict(
    model_name = MODEL_NAME,
    LEARINGIN_RATE = 3e-5,
    EPOCHS = 5,
    BATCH_SIZE = 32,
    TRAIN_VAL_SPLIT = 0.8,
    num_warmup_steps = 3000,
    CLIP_GRAD_VALUE = 5,
    USE_CLIP_GRAD = True,
    optimizer = "AdamW",
    max_length = 196,
    num_cycles = 2,
    sheculer = 'get_cosine_with_hard_restarts_schedule_with_warmup'
)


files_configs = dict(
    train_path = "./public_data/MULTI_Multilingual/multi_train.conll",
    test_path = "./public_data/MULTI_Multilingual/multi_dev.conll",

    wandb_run_name = "xlm-roberta-largV3",
    wandb_notes = "xlm-roberta-larg",
    base_model_path = "./xlm-roberta-larg",
    res_path = "xlm_roberta_larg_for_multi_ner_v3"
)
