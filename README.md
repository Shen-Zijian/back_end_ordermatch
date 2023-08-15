# Order-Driver Matching Service API Documentation

## Overview

This document describes the interfaces for the Order-Driver Matching Service, a backend module developed with the Flask framework. This service accepts order and driver data from the front end, matches them, and returns the matching results to the front end.

## API Endpoints

### 1. POST /api/match

This endpoint accepts order and driver data and returns the match result.

#### Request
The path of request is:
```python
@app.route('/process_data', methods=['POST'])
```
 http://47.243.58.57:80//process_data

##### Headers

| Name         | Value            |
| --- | ----------- |
| Content-Type | application/json |

##### Body

The request body should contain two parameters and two CSV files, 

| Name         | Value            |
| --- | ----------- |
| method       | broadcasting/dispatch |
| radius       | float |
| driver_info  | CSV with columns[driver_id,longitude,latitude,region] |
| order_info   |  CSV with columns[order_id,origin_lng,origin_lat,reward_units,order_region]  |


#### Response

The response body will contain the match result in JSON format:

```json
[
	{
		"order_id":11261.0,
		"order_region":3.0,
		"order_lat":22.3126106262,
		"order_lng":114.1696014404,
		"driver_id":11261.0,
		"driver_region":4.0,
		"driver_lat":22.3129024506,
		"driver_lng":114.1695175171,
		"radius":0.5
	},
]
```
 
## Contact

If you encounter any problems or have any questions, please contact me.

## Change Log

| Date       | Version | Description     |
| ---------- | ------- | --------------- |
| 2023-08-15 | v1.1    | Initial version |

Please make sure to always use the latest version of the API for optimal results.
