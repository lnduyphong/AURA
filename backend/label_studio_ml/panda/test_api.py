"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json
from model import NewModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                # Your input test data here
                 {
      "id": 6,
      "data": {
        "Unnamed: 0": 1976,
        "label": 1,
        "text": "area woman marries into health insurance"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821059Z",
      "updated_at": "2025-02-19T18:41:59.923254Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 6,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 3,
          "result": [
            {
              "from_name": "sentiment",
              "id": "1a617d95-6688-4796-9d85-2d18a6e89f62",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Positive"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 55 minutes",
          "score": 0.562484860420227,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:41:59.933464Z",
          "updated_at": "2025-02-19T18:41:59.933499Z",
          "model": None,
          "model_run": None,
          "task": 6,
          "project": 1
        }
      ]
    },
    {
      "id": 7,
      "data": {
        "Unnamed: 0": 6600,
        "label": 1,
        "text": "man's ironclad grasp of issue can withstand 2 follow-up questions"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821141Z",
      "updated_at": "2025-02-19T18:45:17.086688Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 7,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 4,
          "result": [
            {
              "from_name": "sentiment",
              "id": "27d51fa4-b95c-44a2-a8a5-0c975e193d30",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Negative"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 52 minutes",
          "score": 0.5664330720901489,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:45:17.096404Z",
          "updated_at": "2025-02-19T18:45:17.096439Z",
          "model": None,
          "model_run": None,
          "task": 7,
          "project": 1
        }
      ]
    },
    {
      "id": 14,
      "data": {
        "Unnamed: 0": 1283,
        "label": 0,
        "text": "11 miserable situations parents know all too well"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821745Z",
      "updated_at": "2025-02-19T05:15:19.821761Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 14,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 15,
      "data": {
        "Unnamed: 0": 1868,
        "label": 1,
        "text": "chess grandmaster tired of people comparing every life situation to chess match"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821825Z",
      "updated_at": "2025-02-19T05:15:19.821839Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 15,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 16,
      "data": {
        "Unnamed: 0": 1066,
        "label": 1,
        "text": "cheney wows sept. 11 commission by drinking glass of water while bush speaks"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821901Z",
      "updated_at": "2025-02-19T05:15:19.821915Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 16,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
        {
      "id": 6,
      "data": {
        "Unnamed: 0": 1976,
        "label": 1,
        "text": "area woman marries into health insurance"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821059Z",
      "updated_at": "2025-02-19T18:41:59.923254Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 6,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 3,
          "result": [
            {
              "from_name": "sentiment",
              "id": "1a617d95-6688-4796-9d85-2d18a6e89f62",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Positive"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 55 minutes",
          "score": 0.562484860420227,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:41:59.933464Z",
          "updated_at": "2025-02-19T18:41:59.933499Z",
          "model": None,
          "model_run": None,
          "task": 6,
          "project": 1
        }
      ]
    },
    {
      "id": 7,
      "data": {
        "Unnamed: 0": 6600,
        "label": 1,
        "text": "man's ironclad grasp of issue can withstand 2 follow-up questions"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821141Z",
      "updated_at": "2025-02-19T18:45:17.086688Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 7,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 4,
          "result": [
            {
              "from_name": "sentiment",
              "id": "27d51fa4-b95c-44a2-a8a5-0c975e193d30",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Negative"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 52 minutes",
          "score": 0.5664330720901489,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:45:17.096404Z",
          "updated_at": "2025-02-19T18:45:17.096439Z",
          "model": None,
          "model_run": None,
          "task": 7,
          "project": 1
        }
      ]
    },
    {
      "id": 14,
      "data": {
        "Unnamed: 0": 1283,
        "label": 0,
        "text": "11 miserable situations parents know all too well"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821745Z",
      "updated_at": "2025-02-19T05:15:19.821761Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 14,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 15,
      "data": {
        "Unnamed: 0": 1868,
        "label": 1,
        "text": "chess grandmaster tired of people comparing every life situation to chess match"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821825Z",
      "updated_at": "2025-02-19T05:15:19.821839Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 15,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 16,
      "data": {
        "Unnamed: 0": 1066,
        "label": 1,
        "text": "cheney wows sept. 11 commission by drinking glass of water while bush speaks"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821901Z",
      "updated_at": "2025-02-19T05:15:19.821915Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 16,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
        {
      "id": 6,
      "data": {
        "Unnamed: 0": 1976,
        "label": 1,
        "text": "area woman marries into health insurance"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821059Z",
      "updated_at": "2025-02-19T18:41:59.923254Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 6,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 3,
          "result": [
            {
              "from_name": "sentiment",
              "id": "1a617d95-6688-4796-9d85-2d18a6e89f62",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Positive"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 55 minutes",
          "score": 0.562484860420227,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:41:59.933464Z",
          "updated_at": "2025-02-19T18:41:59.933499Z",
          "model": None,
          "model_run": None,
          "task": 6,
          "project": 1
        }
      ]
    },
    {
      "id": 7,
      "data": {
        "Unnamed: 0": 6600,
        "label": 1,
        "text": "man's ironclad grasp of issue can withstand 2 follow-up questions"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821141Z",
      "updated_at": "2025-02-19T18:45:17.086688Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 7,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 4,
          "result": [
            {
              "from_name": "sentiment",
              "id": "27d51fa4-b95c-44a2-a8a5-0c975e193d30",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Negative"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 52 minutes",
          "score": 0.5664330720901489,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:45:17.096404Z",
          "updated_at": "2025-02-19T18:45:17.096439Z",
          "model": None,
          "model_run": None,
          "task": 7,
          "project": 1
        }
      ]
    },
    {
      "id": 14,
      "data": {
        "Unnamed: 0": 1283,
        "label": 0,
        "text": "11 miserable situations parents know all too well"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821745Z",
      "updated_at": "2025-02-19T05:15:19.821761Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 14,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 15,
      "data": {
        "Unnamed: 0": 1868,
        "label": 1,
        "text": "chess grandmaster tired of people comparing every life situation to chess match"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821825Z",
      "updated_at": "2025-02-19T05:15:19.821839Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 15,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 16,
      "data": {
        "Unnamed: 0": 1066,
        "label": 1,
        "text": "cheney wows sept. 11 commission by drinking glass of water while bush speaks"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821901Z",
      "updated_at": "2025-02-19T05:15:19.821915Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 16,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
        {
      "id": 6,
      "data": {
        "Unnamed: 0": 1976,
        "label": 1,
        "text": "area woman marries into health insurance"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821059Z",
      "updated_at": "2025-02-19T18:41:59.923254Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 6,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 3,
          "result": [
            {
              "from_name": "sentiment",
              "id": "1a617d95-6688-4796-9d85-2d18a6e89f62",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Positive"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 55 minutes",
          "score": 0.562484860420227,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:41:59.933464Z",
          "updated_at": "2025-02-19T18:41:59.933499Z",
          "model": None,
          "model_run": None,
          "task": 6,
          "project": 1
        }
      ]
    },
    {
      "id": 7,
      "data": {
        "Unnamed: 0": 6600,
        "label": 1,
        "text": "man's ironclad grasp of issue can withstand 2 follow-up questions"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821141Z",
      "updated_at": "2025-02-19T18:45:17.086688Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 7,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 1,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": 1,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": [
        {
          "id": 4,
          "result": [
            {
              "from_name": "sentiment",
              "id": "27d51fa4-b95c-44a2-a8a5-0c975e193d30",
              "to_name": "text",
              "type": "choices",
              "value": {
                "choices": ["Negative"]
              }
            }
          ],
          "model_version": "BertClassifier-v0.0.1",
          "created_ago": "13 hours, 52 minutes",
          "score": 0.5664330720901489,
          "cluster": None,
          "neighbors": None,
          "mislabeling": 0.0,
          "created_at": "2025-02-19T18:45:17.096404Z",
          "updated_at": "2025-02-19T18:45:17.096439Z",
          "model": None,
          "model_run": None,
          "task": 7,
          "project": 1
        }
      ]
    },
    {
      "id": 14,
      "data": {
        "Unnamed: 0": 1283,
        "label": 0,
        "text": "11 miserable situations parents know all too well"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821745Z",
      "updated_at": "2025-02-19T05:15:19.821761Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 14,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 15,
      "data": {
        "Unnamed: 0": 1868,
        "label": 1,
        "text": "chess grandmaster tired of people comparing every life situation to chess match"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821825Z",
      "updated_at": "2025-02-19T05:15:19.821839Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 15,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    },
    {
      "id": 16,
      "data": {
        "Unnamed: 0": 1066,
        "label": 1,
        "text": "cheney wows sept. 11 commission by drinking glass of water while bush speaks"
      },
      "meta": {},
      "created_at": "2025-02-19T05:15:19.821901Z",
      "updated_at": "2025-02-19T05:15:19.821915Z",
      "is_labeled": False,
      "overlap": 1,
      "inner_id": 16,
      "total_annotations": 0,
      "cancelled_annotations": 0,
      "total_predictions": 0,
      "comment_count": 0,
      "unresolved_comment_count": 0,
      "last_comment_updated_at": None,
      "project": 1,
      "updated_by": None,
      "file_upload": 1,
      "comment_authors": [],
      "annotations": [],
      "predictions": []
    }
            }
        }],
        # Your labeling configuration here
      "label_config": "<View>\n  <Text name=\"text\" value=\"$text\"/>\n  <View style=\"box-shadow: 2px 2px 5px #999; padding: 20px; margin-top: 2em; border-radius: 5px;\">\n    <Header value=\"Choose text sentiment\"/>\n    <Choices name=\"sentiment\" toName=\"text\" choice=\"single\" showInLine=\"true\">\n      <Choice value=\"Positive\"/>\n      <Choice value=\"Negative\"/>\n      <Choice value=\"Neutral\"/>\n    </Choices>\n  </View>\n</View><!-- {\n  \"data\": {\"text\": \"This is a great 3D movie that delivers everything almost right in your face.\"}\n} -->",
    }

    expected_response = {
        'results': [{
            # Your expected result here
        }]
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    print(response)
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response == expected_response
