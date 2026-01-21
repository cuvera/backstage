Bot            Gateway          Meeting Service (OWNER)        Google ASR           Client
 |                |                         |                      |                  |
 |-- WS connect ->|                         |                      |                  |
 | (meeting_id)   |--   route by hash -->   |                      |                  |
 |                |                         |-- register bot WS -->|                  |
 |                |                         |                      |                  |
 |==    audio frames =====================> |                      |                  |
 |                |                         |-- stream audio ----->|                  |
 |                |                         |                      |                  |
 |                |                         |<-- partial transcript|                  |
 |                |                         |                      |                  |
 |                |                         |<-- final transcript--|                  |
 |                |                         |                      |                  |
 |                |                         |-- generate analysis  |                  |
 |                |                         |                      |                  |
 |                |<==  WS analysis msg ==> |                      |                  |
 |                |                         |                      |                  |
 |                |                         | ===================== ws analysis =====>|
