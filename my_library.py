def test_load():
  return 'loaded'


def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]


def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor


def cond_probs_product(full_table, evidence_values, target_column_, target_value):
  cond_prob_list = []
  evidence_columns = up_list_column_names(full_table)
  evidence_columns = evidence_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_values)
  
  for item in evidence_complete:
    new_item = cond_prob(full_table, item[0], item[1], target_column_, target_value)  
    cond_prob_list += [new_item]
  
  partial_numerator = up_product(cond_prob_list)  
  return partial_numerator


def prior_prob(full_table, target, target_value):
  t_list = up_get_column(full_table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a 


def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_f0 = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)


  #do same for P(Flu=1|...)
  p_f1 = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p_f0, p_f1)
  
  #return your 2 results in a list
  return [neg, pos]


def metrics(zip_list):
  assert type(zip_list)==list, "Input must be a list"
  assert all(isinstance(item, tuple) and len(item) == 2 for item in zip_list), "Input must be a zipped list of pairs"
  for pair in zip_list:
    assert (isinstance(pair, list) and len(pair) == 2), "Input must be a zipped list"
    assert type(pair[0])==int and type(pair[1])==int, "Each value in the pair must be an int"
    assert pair[0]>=0 and pair[1]>=0, "Each value in the pair must be >=0"
  tn=0
  fn=0
  tp=0
  fp=0
  for pair in zip_list:
    if pair == [0,0]:
      tn+=1
    elif pair == [1,1]:
      tp+=1
    elif pair == [1,0]:
      fp+=1
    elif pair == [0,1]:
      fn+=1
  accuracy = sum([p==a for p,a in zip_list])/len(zip_list)
  precision = (tp/(tp+fp)) if (tp+fp)>0 else 0
  recall = (tp/(tp+fn)) if (tp+fn)>0 else 0
  f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
  return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}


