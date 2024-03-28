'''
Here will create a binary and multi-class synthetic classification datasets and for each will calculate the precision,
recall & F-Measure for each one using their respective sklearn method
'''

from sklearn.metrics import precision_score, recall_score, f1_score

# Define binary classification dataset
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# Define multi-class classification dataset
act_pos1 = [1 for _ in range(100)]
act_pos2 = [2 for _ in range(100)]
act_neg1 = [0 for _ in range(10000)]
mc_true = act_pos1 + act_pos2 + act_neg1

# Calculate precision for 1:100 dataset with 90 tp & 30 fp
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = pred_pos + pred_neg
# Calculate precision
precision = precision_score(y_true, y_pred, average='binary')
print('Binary Precision: %.3f' % precision)

# Calculate precision for 1:1:100 dataset with 50tp, 20fp, 99tp & 51fp
pred_pos1 = [0 for _ in range(50)] + [1 for _ in range(50)]
pred_pos2 = [0 for _ in range(1)] + [2 for _ in range(99)]
pred_neg1 = [1 for _ in range(20)] + [2 for _ in range(51)] + [0 for _ in range(9929)]
y_pred1 = pred_pos1 + pred_pos2 + pred_neg1
# Calculate prediction, for MC define labels & use 'micro' for average
precision_mc = precision_score(mc_true, y_pred1, labels=[1, 2], average='micro')
print('Multi-Class Precision: %.3f' % precision_mc)


# Calculate recall for 1:100 dataset with 90 tp & 10 fn
recall_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
recall_neg = [0 for _ in range(10000)]
recall_pred = recall_pos + recall_neg
# Calculate recall
recall = recall_score(y_true, recall_pred, average='binary')
print('Binary Recall: %.3f' % recall)

# Calculate recall for 1:1:100 dataset with 77 tp, 23 fn, 95 tp & 5 fn
recall_pos1 = [0 for _ in range(23)] + [1 for _ in range(77)]
recall_pos2 = [0 for _ in range(5)] + [2 for _ in range(95)]
recall_neg1 = [0 for _ in range(10000)]
recall_pred_mc = recall_pos1 + recall_pos2 + recall_neg1
# Calculate recall
recall_mc = recall_score(mc_true, recall_pred_mc, labels=[1, 2], average='micro')
print('Multi-Class Recall: %.3f' % recall_mc)

# Calculate F-Measure fo 1:100 dataset with 95 tp, 5 fn & 55 fp
predf_pos = [0 for _ in range(5)] + [1 for _ in range(95)]
predf_neg = [1 for _ in range(55)] + [0 for _ in range(9945)]
predf = predf_pos + predf_neg
# Calculate f-score
score = f1_score(y_true, predf, average='binary')
print('Binary F-Measure: %.3f' % score)

