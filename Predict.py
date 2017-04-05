from sklearn import tree
from IPython.display import Image  
import pydotplus

X = [[0, 1], [1, 0], [100, 101]]
Y = [0, 1, 9]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)



dot_data = tree.export_graphviz(clf, out_file=None, 
						 feature_names=X,
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("iris2.pdf")  

print (clf.predict([[50., 51.]]))

print (clf.predict([[51., 51.]]))