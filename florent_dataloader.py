import tensorflow as tf
import numpy as np
import pandas as pd


def modified_CumSum_data_gen_TS_variable_batch_size(in_ds, out_ds, aug=False, in_ch=13, out_ch=2, batch_size=16,
                                                    paddingvalue=0., maskvalue=-50, mag_sig=True, dropdublct=False,
                                                    hoverings=False, bridge=False, force_thrsh=0.01, diffOp=False,
                                                    filtering=False, down_smapling=False, tsframing=False, win_size=5,
                                                    fltTrsh=0.075, mario_corr=False, batch_cropping=False):
    """
    This function is a data generator that embed shorter time series to the length of the lognest time series
    in the generated batches
    :param in_ds: input sensor time series list
    :param out_ds: output tablet time series list
    :param aug: enable the application of the data-augmentation
    :param in_ch: number of tablet data channels
    :param out_ch: number of sensor data channels
    :param batch_size: number of samples in a batch
    :param paddingvalue: the padding value added to the end of the shorter sensor ts; default value=0.
    :param maskvalue: the mask value added to the end of the shorter tablet ts; default value=-50
    :param mag_sig: enable the magnumeter signals
    :param dropdublct: enable the drop of dublicated points form the the input and outpt ts
    :param hoverings: True: if hovering training otherwise False
    :param bridge: True: if training on whole samples while droping hovering signals from the sensor and table signals
    :param force_thrsh: Define pen-up and pen-down signals threshold on the force sensor signal
    :param diffOp: calculte the first derivative of the tablet signal
    :param filtering: True: filter the input and output samples standing on the ratio of the dublicated points
    (serve to illiminate the samples whose long hoverings). Default value=0.025
    TODO: define the appropriate set of thersholds according to the dataset under study
    :param down_smapling: True: down sample the input and output samples otherwise False
    :param tsframing: True: framing the input samples (used only for training 2DCNN_TCN model) otherwise False
    :param win_size: frame_size/2-1 (used when tsframing is True). default value = 5 (create frame_size = 11)
    :return: tuple of input and output batches
    """
    from sklearn.utils import shuffle
    from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise, Convolve, Resize, Pool, Dropout, utils, tsFraming
    import tsaug
    import random

    IN_CHANNELS = in_ch
    OUT_CHANNELS = out_ch
    c = 0

    # if hoverings:
    #     feature_list_in = in_ds[0].columns.values
    #     feature_list_out = ['x', 'y']
    # else:
    #     feature_list_in = in_ds[0].columns.values
    #     feature_list_out = out_ds[0].columns.values
    feature_list_in = in_ds[0].columns.values
    feature_list_out = ['x', 'y']

    if mario_corr:
        print(f"**************************************\n")
        print(f" Apply Mario Orientation corrections  \n")
        print(f"**************************************\n")
        import warnings
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        MAX_FORCE = 4096
        feature_list_in = ["RAW_ACC_FRONT_X", "RAW_ACC_FRONT_Y", "RAW_ACC_FRONT_Z", "RAW_ACC_REAR_X", "RAW_ACC_REAR_Y", "RAW_ACC_REAR_Z", "RAW_GYRO_X", "RAW_GYRO_Y", "RAW_GYRO_Z", "RAW_MAG_X", "RAW_MAG_Y", "RAW_MAG_Z", "RAW_FORCE_TIP"]
       #  raw_clmn = ['MILLIS', 'RAW_ACC_FRONT_X', 'RAW_ACC_FRONT_Y', 'RAW_ACC_FRONT_Z',
       # 'RAW_ACC_REAR_X', 'RAW_ACC_REAR_Y', 'RAW_ACC_REAR_Z', 'RAW_GYRO_X',
       # 'RAW_GYRO_Y', 'RAW_GYRO_Z', 'RAW_MAG_X', 'RAW_MAG_Y', 'RAW_MAG_Z',
       # 'RAW_FORCE_TIP', 'RAW_FORCE_GRIP', 'COUNTER']
        for f, s in enumerate(in_ds):
            in_ds[f] = inscale_1_by_1(in_ds[f])
            in_ds[f].columns = feature_list_in
            in_ds[f] = apply_sigrot(in_ds[f])
            if len(in_ds[f])==0:
                print(f'WROOOOONG')
                break
            in_ds[f]['RAW_FORCE_TIP'] = in_ds[f]['RAW_FORCE_TIP'] / MAX_FORCE
            if in_ds[f].isnull().values.any():
                print(f"ERRRRRRRORRRRR")
                break
            if len(in_ds[f])==0:
                print(f'WROOOOONG')
                break



    if diffOp:
        for u, o in enumerate(out_ds):
            out_ds[u] = out_ds[u].cumsum(axis=0)
            out_ds[u] = out_ds[u].iloc[1, :] - out_ds[u].iloc[0, :]

    if dropdublct:
        for u, o in enumerate(in_ds):
            df = pd.concat([in_ds[u].reset_index(drop=True), out_ds[u].reset_index(drop=True)], axis=1,
                           ignore_index=False, sort=False).reset_index(drop=True)
            fil = df.groupby('x').filter(lambda e: len(e) <= 4)
            if fil.shape[0] != 0:
                out_ds[u] = fil[feature_list_out]
                in_ds[u] = fil[feature_list_in]

    if filtering:
        ii = 0
        cnt = 0
        print(f"in_ds len : {str(len(in_ds))}")
        for u, o in enumerate(out_ds):
            df = out_ds[u].groupby(out_ds[u].columns.tolist(), as_index=False).size()

            if (df['x'] == 0).sum()/len(df) <= fltTrsh:
                in_ds[ii] = in_ds[u]
                out_ds[ii] = out_ds[u]
                ii += 1
            else:
                in_ds.pop(u)
                out_ds.pop(u)
                cnt += 1
                continue
        print(f">>>>>> CNT over hoverings: {cnt}; #of samples : {str(len(in_ds))}")

    if down_smapling:
        print("======================================\n")
        print("           down-sampling              \n")
        print("======================================\n")

        for u, o in enumerate(in_ds):

            df = pd.concat([in_ds[u].reset_index(drop=True), out_ds[u].reset_index(drop=True)], axis=1,
                           ignore_index=False, sort=False).reset_index(drop=True)
            df = df.drop_duplicates(['x', 'y'], keep="last")

            if df.shape[0] != 0:
                out_ds[u] = df[feature_list_out] #.diff().fillna(0)#.drop(index=df.index[0], axis=0, inplace=True)
                in_ds[u] = df[feature_list_in]
            if pd.isna(df[feature_list_out].any()).any() or pd.isna(df[feature_list_in].any()).any():
                print(pd.isna(df[feature_list_out]).any())

    if bridge:
        print("======================================\n")
        print("           Bridging                   \n")
        print("======================================\n")
        in_ds, out_ds = bridging(in_ds, out_ds, force_thrsh=0.01)


    while (True):
        TIME_STEPS, SHORT_STEPS = 0, 100000000
        for p, di in enumerate(in_ds[c:c + batch_size]):
            if len(di) == 0:
                continue
            if TIME_STEPS <= len(di):
                TIME_STEPS = len(di)
            if SHORT_STEPS > len(di) and len(di) != 0:
                SHORT_STEPS = len(di)


        if hoverings:
            in_ = np.zeros((batch_size, TIME_STEPS, IN_CHANNELS)).astype('float')
            out_ = np.zeros((batch_size, OUT_CHANNELS)).astype('float')
        elif batch_cropping:
            in_ = np.zeros((batch_size, SHORT_STEPS, IN_CHANNELS)).astype('float')
            out_ = np.zeros((batch_size, SHORT_STEPS, OUT_CHANNELS)).astype('float')
        else:
            in_ = np.zeros((batch_size, TIME_STEPS, IN_CHANNELS)).astype('float')
            out_ = np.zeros((batch_size, TIME_STEPS, OUT_CHANNELS)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            if len(in_ds[i]) == 0 or len(out_ds[i]) == 0:
                continue
            if batch_cropping:
                if len(in_ds[i]) > SHORT_STEPS:
                    train_x = in_ds[i].iloc[:SHORT_STEPS, :]
                elif len(in_ds[i]) != 0:
                    train_x = in_ds[i]
                else:
                    continue

            elif len(in_ds[i]) < TIME_STEPS:
                pad_in = pd.DataFrame(paddingvalue, index=np.arange(int(TIME_STEPS - len(in_ds[i]))),
                                      columns=feature_list_in)
                train_x = pd.concat([pad_in, in_ds[i]])  # .to_numpy()
            else:
                train_x = in_ds[i]  # .to_numpy()

            train_x = tf.convert_to_tensor(train_x, dtype=float32)
            if batch_cropping:
                if len(out_ds[i]) > SHORT_STEPS:
                    train_y = out_ds[i].iloc[:SHORT_STEPS, :]
                else:
                    train_y = out_ds[i]
            elif len(out_ds[i]) < TIME_STEPS: # and not hoverings:

                pad_out = pd.DataFrame(maskvalue, index=np.arange(int(TIME_STEPS - len(out_ds[i]))),
                                       columns=feature_list_out)
                train_y = pd.concat([out_ds[i], pad_out])
            else:
                train_y = out_ds[i]


            train_y = tf.convert_to_tensor(out_ds[i], dtype=float32)
            if train_x.shape[0] == train_y.shape[0] and train_x.shape[0] != 0 and train_y.shape[0] != 0 or hoverings:
                in_[i - c] = train_x #tf.convert_to_tensor(train_x, dtype=float32)  # add to array - img[0], img[1], and so on.
                out_[i - c] = train_y #tf.convert_to_tensor(train_y, dtype=float32)
            else:
                print(np.shape(in_[i - c]), "+++++++++++", train_x.shape, "+++++++++++++++++", train_y.shape, "xxxxxx",TIME_STEPS)
                continue

        c += batch_size
        if (c + batch_size >= len(in_ds)):
            c = 0
        n = random.randrange(1, 8, 1)
        r = n
        if aug == True and r % 2 == 0 and TIME_STEPS // n > 50:
            my_augmenter = (
                Crop(size=int(int(int(TIME_STEPS // n))))  # random crop subsequences with length 19
            )
            in_, out_ = my_augmenter.augment(in_, out_)

        if mag_sig:
            if tsframing:
                in_ = tsFraming(in_, win_size, IN_CHANNELS)
            yield in_, out_  # [in_[:, :, 0:3], in_[:, :, 3:6], in_[:, :, 6:9], in_[:, :, 9:12],
        else:
            if tsframing:
                in_ = tsFraming(in_, win_size, IN_CHANNELS)
            yield in_, out_  # [in_[:, :, 0:3], in_[:, :, 3:6], in_[:, :, 6:9],