from utils.text_preprocessing import preprocess_amazoncat13k

preprocess_amazoncat13k('trn')
preprocess_amazoncat13k('tst')

# preprocess_amazoncat13k('trn', append_content=True)
# preprocess_amazoncat13k('tst', append_content=True)
