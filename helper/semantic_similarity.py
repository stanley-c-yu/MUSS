from requests import get
import numpy as np
import asyncio
import concurrent.futures
import pandas as pd

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"


def get_semantic_similarity(s1, s2, t='relation', corpus='webbase'):
    try:
        response = get(sss_url, params={
                       'operation': 'api', 'phrase1': s1, 'phrase2': s2, 'type': t, 'corpus': corpus})
        return float(response.text.strip())
    except:
        print('Error in getting similarity for %s: %s' % ((s1, s2), response))
        return 0.0


def get_most_similar(s1, s2_list, t='relation', corpus='webbase'):
    scores = [get_semantic_similarity(
        s1, s2, t=t, corpus=corpus) for s2 in s2_list]
    return s2_list[np.argmax(scores)]


def get_most_similar_batch(s1_list, s2_list, t='relation', corpus='webbase'):
    async def batch_requests():
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor,
                    get_most_similar,
                    s1, s2_list, t, corpus
                )
                for s1 in s1_list
            ]
            responses = [response for response in await asyncio.gather(*futures)]
        return responses

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(batch_requests())
    return result
