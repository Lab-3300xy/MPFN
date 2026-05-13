import argparse

parser = argparse.ArgumentParser(description='Time Series prediction')

parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")

parser.add_argument('--data_type', type=str, default='custom', help='dataset type')
parser.add_argument('--data_name', type=str, default='weather', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/weather/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weather.csv', help='data csv file')

parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')

parser.add_argument('--feature_size', type=int, default=21, help='feature size')

parser.add_argument('--encoder_num', type=int, default=2, help='num of encoder layers')
parser.add_argument('--cross_num', type=int, default=1, help='num of cross-branch')
parser.add_argument('--intra_num', type=int, default=1, help='num of intra-branch')
parser.add_argument('--embed_size', type=int, default=64, help='dimension of model')

parser.add_argument('--forward_expansion', type=int, default=4, help='forward_expansion')
parser.add_argument('--patch_kernel_size', type=int, default=7, help='kernel size for patch')
parser.add_argument('--patch_stride', type=int, default=5, help='stride for patch')
parser.add_argument('--conv_kernel_size', type=int, default=3, help='kernel size for conv')

parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--dropout1', type=float, default=0.1, help='dropout1')

parser.add_argument('--embed_tf_coding', type=str, default='timeF', help='embed_tf_coding')
parser.add_argument('--freq', type=str, default='h', help='freq')
parser.add_argument('--tf_task', type=str, default='M', help='tf_mask')
parser.add_argument('--target', type=str, default='OT', help='target')
parser.add_argument('--adjust_type', type=str, default='TST', help='adjust_type')

args = parser.parse_args()