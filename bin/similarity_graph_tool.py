import networkx as nx
from tqdm.auto import tqdm

def create_graph(matrix, label = None) :
    """
    Function to generate and export a graph from a similarity matrix. 
    If data have differents individuals a column of label can be provided for a better visualization.
    """
    G = nx.Graph()
    #metabolite = matrix.columns
    shape0 = matrix.shape[0]
    shape1 = matrix.shape[1]
    for index in range(shape1) :
        if type(label) == type(None) :
            G.add_node(index)
        else :
            G.add_node(index, disease = label[index]) # Assign label to the node to identify wich disease it is
        for adjancy in range(index + 1, shape1) :
            G.add_edge(index,adjancy,weight = 0) # Initialize all edges and set weight to 0

    for columnidx in tqdm(range(shape0)):
        for subjectaidx in range(shape1):
            for subjectbidx in range(subjectaidx + 1,shape1):
                G[subjectaidx][subjectbidx]["weight"] += matrix[columnidx][subjectaidx][subjectbidx] # Reajust weight according to the metrics calculate in matrix

    #nx.draw(G, with_labels=True, font_weight='bold')shape0 = dataset.shape[0] # For an internal visualisation

    nx.write_gml(G,"generated_graph.gml") # Export graph to file
    print("Graph has been generated in cwd")