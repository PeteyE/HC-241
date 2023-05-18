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



