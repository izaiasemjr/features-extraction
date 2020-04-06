## FUNCTIONS #####################################
comparePerson <- function(p1, p2){
 
  pl1 = strsplit(toString(p1),'_')[[1]][1]
  pl2 = strsplit(toString(p2),'_')[[1]][1]
  
  if(pl1==pl2)
    return (TRUE)
  else
    return (FALSE)  
  
}
##################################################


# M = read.csv(file = "Documentos/MESTRADO/teste-features/data/metrics-extraction/metrics.dat")
personTest = unique(M[,'source'])
c = 0

for ( p in personTest) {
  metricsP = M[M$source== p ,] 
  # pega o que teve mais matches
  maxFit = max(metricsP[,'nfc'])
    
  maxs = metricsP[metricsP[,'nfc']== maxFit,]
  
  
  if(comparePerson(maxs[1,]$source,maxs[1,]$target)) {
    c=c+1
  }
  
  print(maxs)
  print('')
  
  

}



