from keras import backend as K
from keras.models import model_from_json
from training.libs.text_processing import *


with open('training/libs/config.yaml') as f:
    configs = yaml.load(f, Loader=Loader)

path_model_result = 'model_result'

template_logs = '$date - [$type] : resource - $source, Data - [$kind]: $msg\n'


def get_init_logging(name_file = 'some_log'):
    """
    logging
    """

    logger = CustomLOG(
        format_str = FORMAT,
        log_path = os.path.join(os.getcwd(), f'{name_file}.log'),
        resource = name_file,
                      )
    return logger


def get_prediction(language_ = 'rus', names_list = []):
    """
    get dictionary with predictions of gender
    """
    
    if names_list == []: return {}

    model_path = [os.path.join(path_model_result, x) for x in os.listdir(path_model_result) if language_ in x and '.json' in x][0]
    weights_path = [os.path.join(path_model_result, x) for x in os.listdir(path_model_result) if language_ in x and '.h5' in x][0]
    count_names = len(names_list)

    ### initialization model
    K.set_learning_phase(0)
    with open(model_path, 'r') as fp:
        model = model_from_json(fp.read())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #### weights initialization
    model.load_weights(weights_path)

    txtpr = TextProcess(language_letters = configs[language_]['letters'], \
                        gender_dict = configs[language_]['genders'], 
                        params = {}, 
                        loggingpath = os.getcwd(),
                        size_patch = count_names)
    
    txtpr.logger.info(f"Prediction for {language_}, {count_names} count namesets, model version - {model_path.split('_')[-2]}",\
                      kind = 'model prediction')
    result = txtpr.make_predict(model, names_list)

    return result