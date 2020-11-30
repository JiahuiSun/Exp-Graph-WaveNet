# python train.py --device cuda  \
#     --gcn_bool --adjtype doubletransition \
#     --addaptadj  --randomadj  --epoch 80 --expid 5  \
#     --data final_data/shenzhen/outflow45 --save ./experiment/shenzhen/outflow45 \
#     --num_nodes 165  \
#     --adjdata final_data/shenzhen/shenzhen_adj_mx.csv

ep=100
dv=cuda:1
expid=3

python train.py --device $dv --adjtype transition --epoch $ep --expid ${expid} \
                --gcn_bool --aptonly  --addaptadj --randomadj  \
                --data final_data/shenzhen/outflow30 \
                --save ./experiment/shenzhen/outflow30 \
                --num_nodes 165 \
                --adjdata final_data/shenzhen/shenzhen_adj_mx.csv

# > ./experiment/metr/train-$expid.log
# rm ./experiment/metr/metr*
