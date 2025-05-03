# Training data creation

## CASE 1: Retain WHERE clause

```json
{
  "question": "What is the average duration of my runs?",
  "sql_query": "SELECT AVG(duration) AS avg_run_duration FROM workout WHERE workout_type = 'run';",
  "answers": [
    49.34
  ],
  "id": "train_persona_9-question_1199",
  "q_id": 1199,
  "original_persona": "persona_21",
  "reference_date": "2024-11-25",
  "retrieve_calls": [
    "RETRIEVE(query=\"I went running\")"
  ],
  "retrieval_sql_query": "SELECT id FROM workout WHERE workout_type = 'run';",
  "persona": "train_persona_9",
  "operator_trees": [
    {
      "qu_branch_input": null,
      "qu_input": "{{ QU(question=\"What is the average duration of my runs?\") }}",
      "childs": [
        {
          "qu_branch_input": "QU(question=\"What is the average duration of my runs?\")",
          "qu_input": "AVG(l={{ QU(question=\"my runs with duration\") }}, attr_name=\"duration\")",
          "childs": [
            {
              "qu_branch_input": "QU(question=\"my runs with duration\")",
              "qu_input": "SELECT(l={{ QU(question=\"I went running\") }}, attr_names=[\"duration\"], attr_types=[float])",
              "childs": [
                {
                  "qu_branch_input": "QU(question=\"I went running\")",
                  "qu_input": "RETRIEVE(query=\"I went running\")",
                  "childs": [
                    
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

______________________________________________________________________________________________________

## CASE 2: Drop WHERE clause (str-match)

```json
{
  "question": "How many times did I watch an episode of Caprica?",
  "sql_query": "SELECT COUNT(*) AS watch_count FROM tvseries_stream WHERE tvseries_title = 'Caprica';",
  "answers": [
    20
  ],
  "id": "train_persona_9-question_1061",
  "q_id": 1061,
  "original_persona": "persona_21",
  "reference_date": "2024-11-25",
  "retrieve_calls": [
    "RETRIEVE(query=\"I watched a TV series\")"
  ],
  "retrieval_sql_query": "SELECT id FROM tvseries_stream WHERE tvseries_title = 'Caprica';",
  "persona": "train_persona_9",
  "operator_trees": [
    {
      "qu_branch_input": null,
      "qu_input": "{{ QU(question=\"How many times did I watch an episode of Caprica?\") }}",
      "childs": [
        {
          "qu_branch_input": "QU(question=\"How many times did I watch an episode of Caprica?\")",
          "qu_input": "SUM(l={{ QU(question=\"episodes of Caprica I watched\") }}, attr_name=\"duration\")",
          "childs": [
            {
              "qu_branch_input": "QU(question=\"episodes of Caprica I watched\")",
              "qu_input": "SELECT(l={{ QU(question=\"I watched a TV series\") }}, attr_names=[\"episode_name\", \"duration\"], attr_types=[str, float])",
              "childs": [
                {
                  "qu_branch_input": "QU(question=\"I watched a TV series\")",
                  "qu_input": "FILTER(l={{ QU(question=\"I watched a TV series with TV series name\") }}, filter=lambda attr: attr[\"tv_series_title\"].lower().startswith(\"caprica\"))",
                  "childs": [
                    {
                      "qu_branch_input": "QU(question=\"I watched a TV series with TV series name\")",
                      "qu_input": "SELECT(l={{ QU(question=\"I watched a TV series\") }}, attr_names=[\"tv_series_title\"], attr_types=[str])",
                      "childs": [
                        {
                          "qu_branch_input": "QU(question=\"I watched a TV series\")",
                          "qu_input": "RETRIEVE(query=\"I watched a TV series\")",
                          "childs": [
                            
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "qu_branch_input": null,
      "qu_input": "{{ QU(question=\"How many times did I watch an episode of Caprica?\") }}",
      "childs": [
        {
          "qu_branch_input": "QU(question=\"How many times did I watch an episode of Caprica?\")",
          "qu_input": "SUM(l={{ QU(question=\"episodes of Caprica I watched\") }}, attr_name=\"duration\")",
          "childs": [
            {
              "qu_branch_input": "QU(question=\"episodes of Caprica I watched\")",
              "qu_input": "SELECT(l={{ QU(question=\"I watched a TV series\") }}, attr_names=[\"episode_name\", \"duration\"], attr_types=[str, float])",
              "childs": [
                {
                  "qu_branch_input": "QU(question=\"I watched a TV series\")",
                  "qu_input": "FILTER(l={{ QU(question=\"I watched a TV series with TV series name\") }}, filter=lambda attr: attr[\"tv_series_title\"].lower() == \"caprica\")",
                  "childs": [
                    {
                      "qu_branch_input": "QU(question=\"I watched a TV series with TV series name\")",
                      "qu_input": "SELECT(l={{ QU(question=\"I watched a TV series\") }}, attr_names=[\"tv_series_title\"], attr_types=[str])",
                      "childs": [
                        {
                          "qu_branch_input": "QU(question=\"I watched a TV series\")",
                          "qu_input": "RETRIEVE(query=\"I watched a TV series\")",
                          "childs": [
                            
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
______________________________________________________________________________________________________

## CASE 3: Dropped because there are multiple WHERE clauses


```json
{
  "question": "What is the average speed of my runs in 2021?",
  "sql_query": "SELECT AVG(average_speed) AS avg_run_speed FROM workout WHERE workout_type = 'run' AND EXTRACT(YEAR FROM start_date::DATE) = 2021;",
  "answers": [
    10.57
  ],
  "id": "train_persona_9-question_1189",
  "q_id": 1189,
  "original_persona": "persona_21",
  "reference_date": "2024-11-25",
  "retrieve_calls": [
    "RETRIEVE(query=\"I went running\")"
  ],
  "retrieval_sql_query": "SELECT id FROM start_date::DATE) = 2021;",
  "persona": "train_persona_9",
  "operator_trees": [
    {
      "qu_branch_input": null,
      "qu_input": "{{ QU(question=\"What is the average speed of my runs in 2021?\") }}",
      "childs": [
        {
          "qu_branch_input": "QU(question=\"What is the average speed of my runs in 2021?\")",
          "qu_input": "AVG(l={{ QU(question=\"my runs in 2021 with speed\") }}, attr_name=\"speed\")",
          "childs": [
            {
              "qu_branch_input": "QU(question=\"my runs in 2021 with speed\")",
              "qu_input": "SELECT(l={{ QU(question=\"my runs in 2021\") }}, attr_names=[\"speed\"], attr_types=[float])",
              "childs": [
                {
                  "qu_branch_input": "QU(question=\"my runs in 2021\")",
                  "qu_input": "FILTER(l={{ QU(question=\"my runs with date\") }}, filter=lambda attr: attr[\"start_date\"].year == 2021)",
                  "childs": [
                    {
                      "qu_branch_input": "QU(question=\"my runs with date\")",
                      "qu_input": "SELECT(l={{ QU(question=\"my runs\") }}, attr_names=[\"start_date\"], attr_types=[date.fromisoformat])",
                      "childs": [
                        {
                          "qu_branch_input": "QU(question=\"my runs\")",
                          "qu_input": "RETRIEVE(query=\"I went running\")",
                          "childs": [
                            
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
