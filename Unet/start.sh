python main.py --exp 'prostate' --initial_lr 0.001 --contour None --experiment_name "bz12_Dice_loss_prostate_epoch150" --model_path None

python main.py --exp 'liver' --initial_lr 0.0001 --contour "active" --experiment_name "bz12_Active_loss_liver_epoch150" --model_path "/root/chujiajia/Results/bz12_Dice_loss_liver_epoch150/CP_latest9236.pth"


"/root/chujiajia/Results/dice_loss_liver_1/CP_Best.pth"

python main.py --exp 'prostate' --initial_lr 0.0001 --contour "active" --experiment_name "bz12_Prostate_loss_liver_epoch150" --model_path "/root/chujiajia/Results/bz12_Dice_loss_prostate_epoch150/CP_Best.pth"


python main.py --exp 'prostate' --initial_lr 0.001 --contour "None" --experiment_name "bz12_Prostate_dice_loss_epoch150" --model_path "None"


# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--prostatedataroot', default="/root/chujiajia/DataSet/Prostate/")
# parser.add_argument('--liverdataroot', default="/root/chujiajia/DataSet/CHAOS/")
# parser.add_argument('--batchsize', type=int, default=12)
# parser.add_argument('--num_epochs', type=int, default=200)
# parser.add_argument('--gpu', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# parser.add_argument('--exp', type=str, default='prostate')
# parser.add_argument('--initial_lr', type=float, default=0.0001)
# parser.add_argument('--resultpath', type=str, default="/root/chujiajia/Results/")
# parser.add_argument('--experiment_name', type=str,default='bz12_active_loss_prostate_epoch200')
# parser.add_argument('--num_workers', default=8, type=int)
# parser.add_argument('--model_path', type=str,default=None)
# parser.add_argument('--model_path', type=str,default="/root/chujiajia/Results/dice_loss_liver_1/CP_Best.pth")
# parser.add_argument('--model_path', type=str,default="/root/chujiajia/Results/dice_loss_prostate_1/CP_Best.pth")





python main_acdc.py --loss_term "contour" --w_contour 0.1 --experiment_name "Weight_01_contour_loss_acdc_1"
python main_acdc.py --loss_term "contour" --w_contour 0.5 --experiment_name "Weight_05_contour_loss_acdc_1"
python main_acdc.py --loss_term "contour" --w_contour 1.0 --experiment_name "Weight_1_contour_loss_acdc_1"
python main_acdc.py --loss_term "contour" --w_contour 2 --experiment_name "Weight_2_contour_loss_acdc_1"
python main_acdc.py --loss_term "contour" --w_contour 5 --experiment_name "Weight_5_contour_loss_acdc_1"
python main_acdc.py --loss_term "contour" --w_contour 10 --experiment_name "Weight_10_contour_loss_acdc_1"

# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_1"
# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_2"
# python main_acdc.py --loss_term "edge" --experiment_name "1new_lr_edge_loss_acdc_1"
# python main_acdc.py --loss_term "boundary" --experiment_name "1new_lr_boundary_loss_acdc_1"
# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_3"
# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_4"
# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_5"
# python main_acdc.py --loss_term "edge" --experiment_name "1new_lr_edge_loss_acdc_2"
# python main_acdc.py --loss_term "boundary" --experiment_name "1new_lr_boundary_loss_acdc_2"
# python main_acdc.py --loss_term "contour" --experiment_name "1new_lr_contour_loss_acdc_2"
# python main_acdc.py --loss_term "dice" --experiment_name "new_dice_loss_acdc_3"
# python main_acdc.py --loss_term "edge" --experiment_name "new_edge_loss_acdc_3"
# python main_acdc.py --loss_term "boundary" --experiment_name "new_boundary_loss_acdc_3"
# python main_acdc.py --loss_term "contour" --experiment_name "new_contour_loss_acdc_3"
# python main_acdc.py --loss_term "dice" --experiment_name "new_dice_loss_acdc_3"
# python main_acdc.py --loss_term "edge" --experiment_name "new_edge_loss_acdc_1"
# python main_acdc.py --loss_term "boundary" --experiment_name "new_boundary_loss_acdc_1"
# python main_acdc.py --loss_term "contour" --experiment_name "new_contour_loss_acdc_1"
# python main_acdc.py --loss_term "dice" --experiment_name "new_dice_loss_acdc_2"
# python main.py --loss_term "edge" --exp 'prostate' --experiment_name "edge_loss_prostate_1"