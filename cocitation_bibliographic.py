# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:21:59 2017

@author: jhak5
"""

import pandas as pd
import numpy as np

path="D:/jeonhak/platooning/raw/"
ref_info=pd.read_csv(path+'RES_reference_info.csv', encoding='cp437')

au_ti=ref_info[['f-author_year','title']]
ref_id=ref_info['ref_id'].values.tolist()
new_reflist=[]
for i in range(len(au_ti)):
    new_ref=' '.join(map(str,au_ti.ix[i]))
    new_reflist.append(new_ref)
    
doc_id=ref_info['raw_doc'].values.tolist()
#new_dict=dict(zip(new_reflist,ref_id))

new_references=pd.DataFrame({'ref_title':new_reflist,
                             'ref_id':ref_id,
                             'doc_id':doc_id})

doc_list=[]
for i in doc_id:
    if i not in doc_list:
        doc_list.append(i)

ref_list=[]
for i in doc_list:
    ref=new_references.loc[new_references['doc_id']==i]['ref_id'].values.tolist()
    ref=' '.join(map(str,ref))
    ref_list.append(ref)

###############################################################################
def association_strength(symmetric_matrix):
    count_list=[]
    for i in range(len(symmetric_matrix)):
        values=[]
        row_sum=np.sum(symmetric_matrix[i,:])
        for j in range(len(symmetric_matrix)):
            col_sum=np.sum(symmetric_matrix[:,j])
            a=symmetric_matrix[i,j]/(row_sum*col_sum)
            a=str(a)
            values.append(a)
        count_list.append(values)
    asso_matrix=np.asarray(count_list)
    return asso_matrix
###############################################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

count_vec=CountVectorizer()
count_ref=count_vec.fit_transform(ref_list)

count=pd.DataFrame(count_ref.toarray())
count.columns=count_vec.get_feature_names()
t_count=np.transpose(count)

symmetric_matrix=np.dot(t_count,count)
cited_num=symmetric_matrix.diagonal()

asso_matrix=association_strength(symmetric_matrix)
##association strength                                
#cosine_matrix=cosine_similarity(trans_c)#cosine similarity
np.fill_diagonal(asso_matrix,0)                     
asso_matrix=np.tril(asso_matrix,0)                  
class_matrix=pd.DataFrame(asso_matrix)              

ref_name=count_vec.get_feature_names()

class_matrix=class_matrix.convert_objects(convert_numeric=True)
class_matrix=class_matrix.fillna(0)
from scipy import sparse
b=sparse.csr_matrix(class_matrix)
bt=b.tocoo()
row_list=[]
col_list=[]
for i in bt.row:
    c=ref_name[i]
    row_list.append(c)
for i in bt.col:
    c=ref_name[i]
    col_list.append(c)
edge_df=pd.DataFrame({'source':row_list,#bt.row
                      'target':col_list,#bt.col
                      'weight':bt.data})
edge_df['type']='undirected'

new_references.columns.values[1]='id'
node_df=pd.DataFrame({'id':ref_name,
                      'cited_num':cited_num})
node_df2=pd.merge(node_df,new_references,how='inner',on='id')
path="D:/jeonhak/platooning/"
edge_df.to_csv(path+'paper_coci_edge.csv',index=False)
node_df2.to_csv(path+'paper_coci_node.csv',index=False)
############################################################################################
count_vec=CountVectorizer(min_df=1)
count_ref=count_vec.fit_transform(ref_list)

count=np.matrix(count_ref.toarray())
t_count=np.transpose(count)
symmetric_matrix=np.dot(count,t_count)
asso_matrix=association_strength(symmetric_matrix)
##association strength                                
#cosine_matrix=cosine_similarity(trans_c)#cosine similarity
np.fill_diagonal(asso_matrix,0)                     
asso_matrix=np.tril(asso_matrix,0)                  
class_matrix=pd.DataFrame(asso_matrix)              

column_name=pd.DataFrame(doc_list)
f_dir = "F:/2016_project/platooning/raw/"
file = pd.read_csv(f_dir+"filtered_references.csv", encoding='cp437', error_bad_lines=False)
doc_title=file.dropna()
title=doc_title['title'].values.tolist()

class_matrix.columns=doc_list

class_matrix=class_matrix.convert_objects(convert_numeric=True)
class_matrix=class_matrix.fillna(0)
from scipy import sparse
b=sparse.csr_matrix(class_matrix)
bt=b.tocoo()
row_list=[]
col_list=[]
for i in bt.row:
    c=doc_list[i]
    row_list.append(c)
for i in bt.col:
    c=doc_list[i]
    col_list.append(c)
edge_df=pd.DataFrame({'source':row_list,#bt.row
                      'target':col_list,#bt.col
                      'weight':bt.data})

edge_df['type']='undirected'

node_df=pd.DataFrame({'id':doc_list,
                      'label':title})
path="F:/2016_project/platooning/"
edge_df.to_csv(path+'paper_biblio_edge.csv',index=False)
node_df.to_csv(path+'paper_biblio_node.csv',index=False)
