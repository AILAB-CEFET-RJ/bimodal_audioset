#!/bin/bash
if [ -z "$1" ]
then
    echo "Você precisa passar um valor como argumento"
    exit 1
fi

qtd_features=$1
rm -f ${qtd_features}max_subset_unbalanced_train_segments.csv
# Copia os qtd_features primeiros vídeos que tiverem a classe e concatena num csv
for class_name in /m/03m9d0z /m/0ngt1 /m/06mb1 /m/05kq4 /m/02_41 /m/014zdl /m/032s66
do
    grep -m $qtd_features $class_name unbalanced_train_segments.csv >> "${qtd_features}max_subset_unbalanced_train_segments.csv"
done
# Embaralhar os vídeos de cada classe
shuf "${qtd_features}max_subset_unbalanced_train_segments.csv" -o "${qtd_features}max_subset_unbalanced_train_segments.csv"
# Remover linhas duplicadas
awk '!a[$0]++' ${qtd_features}max_subset_unbalanced_train_segments.csv > file.temp && mv file.temp ${qtd_features}max_subset_unbalanced_train_segments.csv

# Adicionar cabeçalho padrão de 3 linhas.
sed -i '1i# YTID, start_seconds, end_seconds, positive_labels' ${qtd_features}max_subset_unbalanced_train_segments.csv
sed -i '1i# num_ytids=Dunno, num_segs=Dunno, num_unique_labels=Dunno, num_positive_labels=Dunno' ${qtd_features}max_subset_unbalanced_train_segments.csv
sed -i '1i# Segments csv adapted from original eval_segments provided by audioset' ${qtd_features}max_subset_unbalanced_train_segments.csv


