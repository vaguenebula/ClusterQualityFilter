import random
import statistics
import threading
import pandas as pd
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
import os
import time
import numpy as np
import dask.dataframe as dd


# Goal: Prune dataset of texts that are not high quality
#############################################################
# Kmeans will cluster texts based on their embeddings
# GPT-4 will evaluate the quality of each cluster
# Pruning will be based on cluster scores and RSD of scores
# RSD should be "low" in order for a cluster evaluation to be considered "reliable"


NUM_OF_CLUSTERS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
kmeans = MiniBatchKMeans(n_clusters=NUM_OF_CLUSTERS, n_init=10, max_iter=1000)
scores = dict()
all_data = dd.read_parquet("./raw_data")

def random_slice(input_string, max_length=500):
    if len(input_string) <= max_length:
        return input_string
    else:
        start_index = random.randint(0, len(input_string) - max_length)
        end_index = start_index + max_length
        return input_string[start_index:end_index]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Returns a numpy array of embeddings for all texts in a parquet file
def get_embeddings(df: pd.DataFrame, limit_samples = None, BATCH_SIZE = 1):
    
    all_embeddings = []
    all_embeddings_df = dd.DataFrame.from_dict({'text': [], 'embeddings': []})
    print(all_data)
    print(all_data.size)
    # Make column in all_data with None values 
    # to be filled with embeddings
    all_data['embeddings'] = None
    for idx, eachText in all_data.iterrows():
        list_of_text = [eachText['text']]
        for i, eachText in enumerate(list_of_text):
            list_of_text[i] = "query: " + eachText

        batch_dict = tokenizer(list_of_text, 
                            max_length=512, padding=True, truncation=True, 
                            return_tensors='pt').to(device)
        outputs = model(**batch_dict)
        embeddings = F.normalize(average_pool(outputs.last_hidden_state, batch_dict['attention_mask']), p=2, dim=1).cpu()
        all_embeddings.append(embeddings.detach())
        all_embeddings_df.append({"text": list_of_text[0], "embeddings": embeddings.detach()})
        del outputs, embeddings, batch_dict
        torch.cuda.empty_cache()

        if limit_samples is not None and idx > limit_samples:
            break
        print(idx)

    all_embeddings_tensor  = torch.cat(all_embeddings, dim=0).numpy()

    return all_embeddings_tensor


# Fit a kmeans model given a Tensor of embeddings
# Saves kmeans model to a pickle file
# min_max_limit is the number of texts to save for the min and max distances of each cluster
def fit_kmeans(embeddings: Tensor):
    kmeans.partial_fit(embeddings)
    

def predict_kmeans(embeddings: Tensor):
    print(kmeans.labels_.size)
    labels = kmeans.predict(embeddings)
    # all_distances = kmeans.transform(embeddings)
    # distances = []
    # for idx, eachLabel in enumerate(labels):
    #     distances.append(all_distances.tolist()[idx][eachLabel])
    
    # distances = pairwise_distances_argmin_min(embeddings, kmeans.cluster_centers_)[0].tolist()
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[kmeans.labels_], axis=1).tolist()

    return labels, distances


def calculate_relative_std_dev_and_mean(data, cluster):

    if not data:
        return None, None

    mean = statistics.mean(data)
    std_dev = statistics.stdev(data)
    relative_std_dev = (std_dev / mean) * 100
    
    scores[cluster] = mean, relative_std_dev


def evaluate_lines(lines: list, cluster: int) -> float:
    import openai

    openai.api_key = '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'
    scores = []

    for eachLine in lines:            
        try:
            if eachLine != "":
                chat_completion = openai.ChatCompletion.create(model="gpt-4", 
                                                            messages=[{"role": "user", "content": f"""You are assigned the task of rating the quality of text data that will be used to train a large language model based on several metrics: high coherence, low repetition, and low noise. Your ratings will then be used to prune a dataset of data that reduces the performance of the model. Please be very strict with your rating.  Only respond with a value between 0 and 1, 1 being the highest possible quality data and 0 being the lowest. 
Data: {random_slice(eachLine)}"""}])
            scores.append(float(chat_completion.choices[0].message.content))
        except Exception as e:
            print(e)
            print(chat_completion.choices[0].message.content)
            continue

    calculate_relative_std_dev_and_mean(scores, cluster)
    print("Average for cluster ", cluster, ": ", scores[cluster][0])



# Evaluate clusters given the furthest and 
# closest texts to the cluster centers
#
# sample limit is the number of texts to evaluate per min and max of the cluster
def evaluate_clusters(sample_limit: int) -> dict:

    threads = []

    for filename in os.listdir("./text_quality"):
        if filename.endswith('.txt'):
            file_path = os.path.join("./text_quality", filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n\n\n")[:sample_limit]
                f.close()
        thread = threading.Thread(target=evaluate_lines, args=(lines, filename.split(".txt")[0]))
        thread.start()
        threads.append(thread)

        # average of 217 tokens per prompt 
        # keeps us under 10k tokens per minute rate limit (i think lol)
        time.sleep(16 * sample_limit / 5)
        
    # Save average score along with RSD for each cluster (GPT-4 scores min and max distance texts)
    try:
        with open('cluster_scores.txt', 'w') as f:
            for key, value in scores.items():
                f.write(str(key) + ' : ' + str(value) + '\n')
    except Exception as e:
        print(e)

def prune_dataset(df: pd.DataFrame, labels: list):

    # Create new dataframe with labels
    data = {'text': df['text'][:len(labels)], 'label': labels}
    df2 = pd.DataFrame(data)
    num_pruned = 0
    scores_dict = dict()

    # Parse scores from file
    with open('cluster_scores.txt', 'r') as f:
        for line in f:
            cluster_number = line.split(':')[0].strip()
            scores_dict[cluster_number] = (float(line.split(':')[1].strip().split(',')[0][1:]), 
                                           float(line.split(':')[1].strip().split(',')[1][:-1]))
    # Prune dataset
    for idx, eachRow in df2.iterrows():
        if scores_dict[eachRow['label']][0] < 0.85:
            df2.drop(idx, inplace=True)
            num_pruned += 1

    # Save pruned dataset
    df2.to_parquet('pruned_data\en_part_00000_pruned.parquet', engine='pyarrow')
    print("Pruned ", num_pruned, " texts")


# Collects embeddings from raw data folder and saves them to a file
def collect_embeddings():
    for eachFile in os.listdir("./raw_data"):
        if eachFile.endswith('.parquet'):
            # df = pd.read_parquet('raw_data/' + eachFile, engine='pyarrow')
            # shuffled_df = df.sample(frac=1, random_state=42)
            shuffled_df = None
            embeddings = get_embeddings(shuffled_df, limit_samples=1000)
            torch.save(embeddings, 'embeddings/' + eachFile.split(".parquet")[0] + '_embeddings.npy')

# finds the min and max distances for each cluster and saves the text to a file
def find_min_max_distances():

    raw_data_files = [i for i in os.listdir("./raw_data") if i.endswith('.parquet')]
    label_files = [i for i in os.listdir("./embeddings/labels") if i.endswith('.txt')]
    distance_files = [i for i in os.listdir("./embeddings/distances") if i.endswith('.txt')]
    min_dict = {i: None for i in range(NUM_OF_CLUSTERS)}
    max_dict = {i: None for i in range(NUM_OF_CLUSTERS)}

    # this could probably be more efficient. my python skills are not the best xD
    for idx, eachFile in enumerate(raw_data_files):
        df = pd.read_parquet('raw_data/' + eachFile, engine='pyarrow')
        with open("./embeddings/labels/" + eachFile.split(".parquet")[0] + "_labels.txt", "r") as f:
            labels = f.read().split("[")[1].split("]")[0].split(", ")
            labels = [int(eachLabel) for eachLabel in labels]
            f.close()
        with open("./embeddings/distances/" + eachFile.split(".parquet")[0] + "_distances.txt", "r") as f:
            distances = f.read().split("[")[1].split("]")[0].split(", ")
            distances = [float(eachDist) for eachDist in distances]
            f.close()
        data = {'text': df['text'][:len(labels)], 'label': labels, 'embedding_distance': distances}
        df2 = pd.DataFrame(data)
        for cluster in range(NUM_OF_CLUSTERS):
            min_list = df2[df2['label'] == cluster].sort_values(by='embedding_distance', ascending=True).head(5)
            max_list = df2[df2['label'] == cluster].sort_values(by='embedding_distance', ascending=True).tail(5)
            print(min_list)
            if min_dict[cluster] is None:
                min_dict[cluster] = min_list
            else:
                min_dict[cluster] = pd.concat([min_dict[cluster], min_list], ignore_index=True).sort_values(by='embedding_distance', ascending=True).head(5)
            if max_dict[cluster] is None:
                max_dict[cluster] = max_list
            else:
                max_dict[cluster] = pd.concat([max_dict[cluster], max_list], ignore_index=True).sort_values(by='embedding_distance', ascending=True).tail(5)

        print(min_dict[0]['embedding_distance'], max_dict[0]['embedding_distance'])

    for eachCluster in max_dict:
        with open("./embeddings/min_max_distances/" + str(eachCluster) + "_min_distances.txt", "w", encoding="utf-8") as f:
            for eachText in min_dict[eachCluster]['text'].tolist():
                f.write(eachText + "\n\n\n")
            f.close()
        with open("./embeddings/min_max_distances/" + str(eachCluster) + "_max_distances.txt", "w", encoding="utf-8") as f:
            for eachText in max_dict[eachCluster]['text'].tolist():
                f.write(eachText + "\n\n\n")
            f.close()
        
def find_clusters():
    df = dd.read_parquet("./raw_data")
    print(df.head())
    # iterates through embeddings to fit kmeans model
    for eachEmbedding in os.listdir("./embeddings"):
        if eachEmbedding.endswith('.pt'):
            embeddings = torch.load("./embeddings/" + eachEmbedding)
            fit_kmeans(embeddings)
    # iterates through embeddings to predict labels and distances
    for eachEmbedding in os.listdir("./embeddings"):
        
        if eachEmbedding.endswith('.pt'):
            embeddings = torch.load("./embeddings/" + eachEmbedding)
            labels, distances = predict_kmeans(embeddings)
            with open("./embeddings/labels/" + eachEmbedding.split("_embeddings.pt")[0] + "_labels.txt", "w") as f:
                f.write(str(labels.tolist()))
                f.close()
            with open("./embeddings/distances/" + eachEmbedding.split("_embeddings.pt")[0] + "_distances.txt", "w") as f:
                f.write(str(distances))
                f.close()

def final(df, labels):
    # (Also saves min and max distances for gpt-4 to evaluate)
    scores = evaluate_clusters(sample_limit=5)
    prune_dataset(df, labels, scores)
    model_filename = './kmeans_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(kmeans, model_file)


collect_embeddings()
find_clusters()
# find_min_max_distances()


