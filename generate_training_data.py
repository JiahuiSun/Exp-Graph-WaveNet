# coding: utf-8
import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        print(type(df.index.values[0]))
        print(df.index.values.astype("datetime64[D]")) 
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        print(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def seq_gen(n_day, data_seq, offset, n_frame, n_route, day_slot, C_0=2):
    '''
    Generate data in the form of standard sequence unit.
    :param n_day: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [n_windows, n_frame, n_route, C_0].
    '''
    # n_day: how many days; data_seq: all datas; offset: start point; n_frame: n_hist+n_pred; day_slot=interval/day
    n_slot = day_slot - n_frame + 1 # how many windows per day
    tmp_seq = np.zeros((n_day * n_slot, n_frame, n_route, C_0))
    for i in range(n_day):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, ...], [n_frame, n_route, C_0])
    return tmp_seq


def generate_train_val_test(args):
    """
    输入的维度是2，[outflow, t]，长度是10
    输出的维度是1，长度是3
    """
    if args.city == 'shenzhen':
        n_train, n_val, n_test = 21, 3, 6
        day_slot = int(16.5 * 60 / args.interval)
    else:
        n_train, n_val, n_test = 17, 3, 5
        day_slot = int(17.5 * 60 / args.interval)
    n_his, n_pred = args.n_his, args.n_pred
    df = pd.read_hdf(args.traffic_df_filename)

    # 0 is the latest observed sample. use last 10 samples to predict
    # x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
    # )
    # x_offsets = np.arange(-(n_his-1), 1, 1)
    # y_offsets = np.sort(np.arange(1, n_pred+1, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    # x, y = generate_graph_seq2seq_io_data(
    #     df,
    #     x_offsets=x_offsets,
    #     y_offsets=y_offsets,
    #     add_time_in_day=True,
    #     add_day_in_week=False,
    # )

    # window slide
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    data_seq = np.concatenate([data, time_in_day], axis=-1)
    print(data_seq.shape)
    seq_train = seq_gen(n_train, data_seq, 0, n_his+n_pred, num_nodes, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_his+n_pred, num_nodes, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_his+n_pred, num_nodes, day_slot)
    
    # split
    x_train, y_train = seq_train[:, :n_his, ...], seq_train[:, n_his:, ...]
    x_val, y_val = seq_val[:, :n_his, ...], seq_val[:, n_his:, ...]
    x_test, y_test = seq_test[:, :n_his, ...], seq_test[:, n_his:, ...]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            # x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            # y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="final_data", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data_processing/processed_data",
        help="Raw traffic readings.",
    )
    parser.add_argument('--n_his', type=int, default=10)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--city', type=str, default='hangzhou')
    parser.add_argument('--interval', type=int, default=10)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f'{args.city}', f'outflow{args.interval}')
    args.traffic_df_filename = os.path.join(args.traffic_df_filename, args.city, f'outflow{args.interval}.h5')
    print(args)

    main(args)
