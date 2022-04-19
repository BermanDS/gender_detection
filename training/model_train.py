from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv1D, MaxPooling1D, Concatenate, LSTM, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from libs.text_processing import *


##### path to data
path_file = os.path.join('data', 'training_dataset_mini.parquet')   # sample of data - about ~140K rows
#path_file = os.path.join('data', 'training_dataset.parquet')       # main data - about 10M rows

#### path to result
path_result = os.path.join('..', 'result')

##### define size of batch according to amount of data
batch_size_ = 9000
#batch_size_ = 90000

##### chosing language 
language_ = 'rus'
#language_ = 'eng'


if __name__ == "__main__":

    ########## loading configs for definite language
    with open('libs/config.yaml') as f:
        configs = yaml.load(f, Loader=Loader)
    configs.keys()
    #####################################################################

    ########## initialization main instance of text proccessing
    txtpr = TextProcess(language_letters = configs[language_]['letters'],
                        gender_dict = configs[language_]['genders'], 
                        params = {}, 
                        loggingpath = os.getcwd(),
                        size_patch = batch_size_)
    #####################################################################

    ######### loading batch of data
    num_batch = 0
    count_batches = pq.read_table(path_file, columns = ['gender']).num_rows // batch_size_
    load, dats, msg = load_parquet_to_array(path_file,\
                                            [('language','=',language_)],\
                                            ['first_name','last_name','middle_name','gender'],\
                                            batch_size_,\
                                            batch_size_ * num_batch,
                                            True)
    ######################################################################

    if not load:
        txtpr.logger.error(f"Problem with loading sample of data for language - {language_}, batch size - {txtpr.size_patch} : {msg}(",\
                          kind = 'preparing data')
    else:
        ###### transforming data for training 
        x_train, x_val, y_train, y_val = txtpr.make_batch(dats)

        ###### Shape of input
        length, depth = len(x_train[0]), len(x_train[0][0])
        length_gender = len(y_train[0])

        ###### initialize input layer
        input_layer = Input(shape=(length, depth), dtype='float32', name='main_input')

        ###### next layers
        conv = Conv1D(256, 5, 
                      strides=1, 
                      padding='same', 
                      dilation_rate=1, 
                      activation='relu', 
                      use_bias=True, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros', 
                      kernel_regularizer=None, 
                      bias_regularizer=None, 
                      activity_regularizer=None, 
                      kernel_constraint=None, 
                      bias_constraint=None)(input_layer)
        
        conv2 = Conv1D(256, 3,
                       strides=1, 
                       padding='same', 
                       dilation_rate=1, 
                       activation='relu',
                       use_bias=True, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='zeros',
                       kernel_regularizer=None, 
                       bias_regularizer=None, 
                       activity_regularizer=None,
                       kernel_constraint=None, 
                       bias_constraint=None)(input_layer)
        
        lstm = LSTM(256)(input_layer)

        added = Concatenate()([conv, conv2])
        
        p = MaxPooling1D(pool_size=4)(added)
        f = Flatten()(p)

        added2 = Concatenate()([f, lstm])

        d = Dropout(0.2, noise_shape=None, seed=None)(added2)
        hidden = Dense(256, activation='relu')(d)
        d = Dropout(0.5, noise_shape=None, seed=None)(hidden)

        sex = Dense(length_gender, kernel_initializer='uniform', activation='softmax', name='sex')(d)

        ##### initialization model
        model = Model(inputs=[input_layer], outputs=[sex])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        while load:
            num_batch += 1
            learn_rate = 0.001 / (num_batch / count_batches)
            txtpr.logger.info(f"{num_batch} batch of data out of {count_batches} for language - {language_}, batch size - {txtpr.size_patch}, learning rate - {learn_rate} ...",\
                              kind = 'training data')
            
            ### tuning model
            opt = Adam(learning_rate = learn_rate)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            
            ### fitting
            model.fit(x_train, y_train, epochs=3, batch_size=10000, validation_data=(x_val, y_val), verbose=2)
            
            ### next batch of data
            load, dats, msg = load_parquet_to_array(path_file,\
                                            [('language','=',language_)],\
                                            ['first_name','last_name','middle_name','gender'],\
                                            batch_size_,\
                                            batch_size_ * num_batch,
                                            True)
            
            if load:
                x_train, x_val, y_train, y_val = txtpr.make_batch(dats)
            else:
                txtpr.logger.error(f"Problem with loading batch of data during fitting for language - {language_}, batch size - {txtpr.size_patch}, {num_batch} batch : {msg}(",\
                                    kind = 'preparing data')
        
        #### saving result of fitted model
        K.set_learning_phase(0)
        with open(os.path.join(path_result, f"{language_}_{datetime.today().strftime('%Y%m%d%H%M')}_model.json"), "w") as json_file:
            json_file.write(model.to_json())
        
        model.save_weights(os.path.join(path_result, f"{language_}_{datetime.today().strftime('%Y%m%d%H%M')}_model.h5"))

        txtpr.logger.info(f"Data of model saved for language - {language_}, count batches - {count_batches}!",\
                            kind = 'saving model')
