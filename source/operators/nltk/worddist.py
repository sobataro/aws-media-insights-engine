# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
import os
import json
from botocore import config
import nltk
import json

from MediaInsightsEngineLambdaHelper import DataPlane
from MediaInsightsEngineLambdaHelper import MediaInsightsOperationHelper
from MediaInsightsEngineLambdaHelper import MasExecutionError

mie_config = json.loads(os.environ['botoConfig'])
config = config.Config(**mie_config)
s3 = boto3.client('s3')


def lambda_handler(event, context):
    print("We got the following event:\n", event)

    operator_object = MediaInsightsOperationHelper(event)

    try:
        bucket = operator_object.input["Media"]["Text"]["S3Bucket"]
        key = operator_object.input["Media"]["Text"]["S3Key"]
    except KeyError as e:
        operator_object.update_workflow_status("Error")
        operator_object.add_workflow_metadata(TranslateError="No valid inputs {e}".format(e=e))
        raise MasExecutionError(operator_object.return_output_object())

    try:
        workflow_id = operator_object.workflow_execution_id
    except KeyError as e:
        operator_object.update_workflow_status("Error")
        operator_object.add_workflow_metadata(TranslateError="Missing a required metadata key {e}".format(e=e))
        raise MasExecutionError(operator_object.return_output_object())

    try:
        asset_id = operator_object.asset_id
    except KeyError:
        print('No asset id for this workflow')
        asset_id = ''

    try:
        s3_response = s3.get_object(Bucket=bucket, Key=key)
        transcribe_metadata = json.loads(s3_response["Body"].read().decode("utf-8"))
        transcript = transcribe_metadata["results"]["transcripts"][0]["transcript"]
    except Exception as e:
        operator_object.update_workflow_status("Error")
        operator_object.add_workflow_metadata(TranslateError="Unable to read transcription from S3: {e}".format(e=str(e)))
        raise MasExecutionError(operator_object.return_output_object())

    nltk.download('punkt', download_dir='/tmp/')
    nltk.data.path.append("/tmp/")
    nltk.data.load('tokenizers/punkt/english.pickle')

    # Split input text into a list of words
    words = nltk.word_tokenize(transcript)
    print("Input text length: " + str(len(transcript)))
    print("Number of words: " + str(len(words)))
    lower_words = [w.lower() for w in words]
    freqdist = nltk.FreqDist(lower_words)

    result = {}
    result["Results"] = dict(freqdist)
    print("FreqDist: " + json.dumps(result["Results"], indent=4))

    dataplane = DataPlane()
    metadata_upload = dataplane.store_asset_metadata(asset_id, operator_object.name, workflow_id, result)
    if "Status" not in metadata_upload:
        operator_object.add_workflow_metadata(
            TranslateError="Unable to upload metadata for asset: {asset}".format(asset=asset_id))
        operator_object.update_workflow_status("Error")
        raise MasExecutionError(operator_object.return_output_object())
    else:
        if metadata_upload['Status'] == 'Success':
            operator_object.update_workflow_status("Complete")
            return operator_object.return_output_object()
        else:
            operator_object.add_workflow_metadata(
                TranslateError="Unable to upload metadata for asset: {asset}".format(asset=asset_id))
            operator_object.update_workflow_status("Error")
            raise MasExecutionError(operator_object.return_output_object())
