
import numpy as np
import random


def sample_change_link(dataset,reduced_dataset,start_step,end_step):
    np.random.seed(42)
    random.seed(42)
    positive_samples = []
    negative_samples = []

    for i in range(start_step-1,end_step):
        graph=dataset[i]
        cur_graph=reduced_dataset[i]
        for e in cur_graph.edges():
            e0_index = dataset.vertex2index[e[0]]
            e1_index = dataset.vertex2index[e[1]]
            positive_samples.append([i, e0_index, e1_index])

    for _ in range(1):
        for i, source_index, target_index in positive_samples:
            graph = dataset[i]
            source = dataset.vertices[source_index]
            target = dataset.vertices[target_index]

            while True:
                if random.randrange(2) == 0: # replace source
                    new_source_index = random.randrange(len(dataset.vertices))
                    new_source = dataset.vertices[new_source_index]
                    if not graph.has_edge(new_source, target):
                        negative_samples.append([i, new_source_index, target_index])
                        break
                else: # replace target
                    new_target_index = random.randrange(len(dataset.vertices))
                    new_target = dataset.vertices[new_target_index]
                    if not graph.has_edge(new_target, source):
                        negative_samples.append([i, source_index, new_target_index])
                        break

    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    samples = np.concatenate((positive_samples, negative_samples))

    positive_labels = np.ones(positive_samples.shape[0])
    negative_labels = np.zeros(negative_samples.shape[0])
    labels = np.concatenate((positive_labels, negative_labels))

    return samples, labels
def sample_change_link_hard(dataset,reduced_dataset,start_step,end_step):
    np.random.seed(42)
    random.seed(42)

    positive_samples = []
    negative_samples = []

    for i in range(start_step-1,end_step):
        graph=dataset[i]
        cur_graph=reduced_dataset[i]
        for e in cur_graph.edges():
            e0_index = dataset.vertex2index[e[0]]
            e1_index = dataset.vertex2index[e[1]]
            positive_samples.append([i, e0_index, e1_index])

        for j in range(i,17):
            next_reduced_graph=reduced_dataset[j+1]
            for e in next_reduced_graph.edges():
                e0_index = dataset.vertex2index[e[0]]
                e1_index = dataset.vertex2index[e[1]]
                if len(negative_samples)>=len(positive_samples):
                    break
                else:
                    negative_samples.append([i, e0_index, e1_index])

        while len(negative_samples)< len(positive_samples):
            new_source_index = random.randrange(len(dataset.vertices))
            new_target_index = random.randrange(len(dataset.vertices))
            new_source = dataset.vertices[new_source_index]
            new_target = dataset.vertices[new_target_index]

            if new_source_index==new_target_index:
                continue
            if not graph.has_edge(new_target, new_source):
                negative_samples.append([i, new_source_index, new_target_index])

    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    samples = np.concatenate((positive_samples, negative_samples))

    positive_labels = np.ones(positive_samples.shape[0])
    negative_labels = np.zeros(negative_samples.shape[0])
    labels = np.concatenate((positive_labels, negative_labels))

    return samples, labels,len(samples)

def sample_all_link(dataset,start_step,end_step):
    np.random.seed(42)
    random.seed(42)

    positive_samples = []
    negative_samples = []


    for i in range(start_step-1,end_step):
        graph=dataset[i]
        for e in graph.edges():
            e0_index = dataset.vertex2index[e[0]]
            e1_index = dataset.vertex2index[e[1]]
            positive_samples.append([i, e0_index, e1_index])

    for _ in range(1):
        for i, source_index, target_index in positive_samples:
            graph = dataset[i]
            source = dataset.vertices[source_index]
            target = dataset.vertices[target_index]

            while True:
                if random.randrange(2) == 0: # replace source
                    new_source_index = random.randrange(len(dataset.vertices))
                    new_source = dataset.vertices[new_source_index]
                    if not graph.has_edge(new_source, target):
                        negative_samples.append([i, new_source_index, target_index])
                        break
                else: # replace target
                    new_target_index = random.randrange(len(dataset.vertices))
                    new_target = dataset.vertices[new_target_index]
                    if not graph.has_edge(new_target, source):
                        negative_samples.append([i, source_index, new_target_index])
                        break

    #print("positive:",len(positive_samples))
    #print("negative:",len(negative_samples))

    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    samples = np.concatenate((positive_samples, negative_samples))

    positive_labels = np.ones(positive_samples.shape[0])
    negative_labels = np.zeros(negative_samples.shape[0])
    labels = np.concatenate((positive_labels, negative_labels))

    return samples, labels

def sample_all_link_hard(dataset,reduced_dataset,start_step,end_step):
    np.random.seed(42)
    random.seed(42)

    positive_samples = []
    negative_samples = []

    for i in range(start_step-1,end_step):
        graph=dataset[i]
        for e in graph.edges():
            e0_index = dataset.vertex2index[e[0]]
            e1_index = dataset.vertex2index[e[1]]
            positive_samples.append([i, e0_index, e1_index])

        next_reduced_graph=reduced_dataset[i+1]

        for e in next_reduced_graph.edges():
            e0_index = dataset.vertex2index[e[0]]
            e1_index = dataset.vertex2index[e[1]]

            if len(negative_samples)>=len(positive_samples):
                break
            else:
                negative_samples.append([i, e0_index, e1_index])

        while len(negative_samples)< len(positive_samples):

            new_source_index = random.randrange(len(dataset.vertices))
            new_target_index = random.randrange(len(dataset.vertices))
            new_source = dataset.vertices[new_source_index]
            new_target = dataset.vertices[new_target_index]

            if new_source_index==new_target_index:
                continue
            if not graph.has_edge(new_target, new_source):
                negative_samples.append([i, new_source_index, new_target_index])

    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    samples = np.concatenate((positive_samples, negative_samples))

    positive_labels = np.ones(positive_samples.shape[0])
    negative_labels = np.zeros(negative_samples.shape[0])
    labels = np.concatenate((positive_labels, negative_labels))
    return samples, labels