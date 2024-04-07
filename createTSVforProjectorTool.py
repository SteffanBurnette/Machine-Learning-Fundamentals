import io #input/output
#Creating an output file for the vectors that represents each of our tokens
def create_tsv(vocabulary, weights):
  ''''
  Creates a tsv file for the embedding projector tool
  '''
  out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
  #Creating an out metadata which is the words of our vocabulary
  out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
  #The code after writes the values 
  for index, word in enumerate(vocabulary):
    if index == 0:
      continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
  out_v.close()
  out_m.close()