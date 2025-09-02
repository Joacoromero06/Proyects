import sys
from collections import defaultdict
class grafo: 
    def __init__(self):
        self.adj=defaultdict(list)
        self.d0=-1
    def agrega_arista(self,u,v,):
        self.adj[u].append((v,1))
        self.adj[v].append((u,1))
    def vecinos(self,v):
        return  self.adj[v]
    def vertices(self):
        return list(self.adj.keys())
    
    def eliminar_ci(self,c_i):
        Gi_sig=grafo()
        for x in self.vertices():
            if x!=c_i:
                for v,w in self.vecinos(x):
                    if v!=c_i:
                        Gi_sig.agrega_arista(x,v)
        Gi_sig.d0=self.d0
        return Gi_sig
    
    #Siempre desde 1 desde la capital
    def dijkstra(self):
        #inicio dijkstra
        V=self.vertices()
        S=set()
        S.add(0)
        inf=1000000000
        n=len(V)
        D=[inf for i in range(n)]
        for v,w in self.vecinos(0):
            D[v]=w#w es igual a 1

        #algoritmo dijkstra    
        while len(S)!=n:
            min= inf
            u=None
            for v in V:
                if v not in S and D[v]<min:
                    u=v
                    min=D[v]
            if u is None:
                break
            S.add(u)

            for v,w in self.vecinos(u):
                if v not in S and D[u]+w<D[v]:
                    D[v]=D[u]+w

        #Resta 1 para camino euleriano
        for i in range(n):
            D[i]-=1
        return D

    def calcula_D0(self):
        dist=self.dijkstra()
        self.d0=dist[len(dist)-1]

    def esta_roto(self):
        Distancias=self.dijkstra()
        if Distancias[len(Distancias)-1]>self.d0:
            return True
        else: 
            return False

    def puntos_criticos(self):
        puntos_criticos=0
        for v in self.vertices():
            g_sig=self.eliminar_ci(v)
            if g_sig.esta_roto():
                puntos_criticos+=1

def resuelve(eliminar,grafo):
    grafo.calcula_D0()
    k=len(eliminar)
    output=list()

    if grafo.esta_roto():
        output.append(-1)
    else:
        output.append(grafo.puntos_criticos())


    for i in range(k):
        c_i=eliminar[i]
        Gi_sig=grafo.eliminar_ci(c_i)
        if Gi_sig.esta_roto():
            l=[-1 for i in range(k)]
            output.append(l)
            break
        else:
            output.append(Gi_sig.puntos_criticos())
    
    return output

def main():
    data= sys.stdin.read().split()
    data= list(map(int,data))
    N,M,K=data[0],data[1],data[2]

    for i in range (3,len(data),1):
        data[i]-=1
        
    Grafonia_0=grafo()
    for i in range(3,2*M +3,2):
        Grafonia_0.agrega_arista(data[i],data[i+1])

    eliminar=list()    
    for i in range(2*M +3,len(data),1):
        eliminar.append(data[i])
    
    salida=resuelve(eliminar,Grafonia_0)
    print(salida)
    for x in salida:
        sys.stdout.write('f{x}\n')

if __name__ =='__main__':
    main()