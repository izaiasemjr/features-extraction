#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:18:24 2020

@author: izaiasjr
"""
import os
import json
import shutil
import numpy as np


def getParamMethode(json, mtype, methode):
    values_param = methode  # methode is the first param
    for (key, values) in json[mtype][methode]['params'].items():
        values_param = values_param + "," + str(values)
    return values_param


def main():
    with open('params.json') as jsonFile:
        paramsJson = json.load(jsonFile)

    rays_norm = [10, 15, 25, 35]
    for radius_norm in rays_norm:

        regions = [
            "nose_tip", "eye_ri", "eye_re", "eye_li", "eye_le", "mouth_r",
            "mouth_l", "mouth_cu", "mouth_cd"
        ]
        for region in regions:

            # rays_feat = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            rays_feat = [30, 40, 50]
            for radius_feat in rays_feat:
                extractors = ['FPFH']
                for extractor in extractors:

                    # Get params from json
                    exe = paramsJson['exe']
                    pathPeople = paramsJson['paths']['clouds']
                    pathOutput = paramsJson['paths']['output']
                    keypointsPath = paramsJson['paths']['keypoints']
                    features = getParamMethode(paramsJson, "features",
                                               extractor)
                    features = '{},{},{}'.format(
                        features.split(',')[0], radius_norm, radius_feat)

                    features_size = paramsJson['features'][extractor][
                        'descriptorSize']

                    # creat path if not exist (by region)
                    pathOutput = '{}/norm_{}/{}'.format(
                        pathOutput, radius_norm, region)
                    if not os.path.exists(pathOutput):
                        os.makedirs(pathOutput)

                    filename = "{}/{}_{}_{}.dat".format(
                        pathOutput, region, radius_feat, extractor)
                    # creating header
                    with open(filename, 'w') as fileoutput:
                        for i in np.arange(1, features_size):
                            fileoutput.write("{},".format(i))
                        fileoutput.write(
                            '{},subject,tp,exp,sample\n'.format(i + 1))
                    fileoutput.close

                    # Separete just neutral faces and aply feature extraction
                    command = ""
                    count = 0
                    for person in os.listdir(pathPeople):
                        pathPerson = pathPeople + "/" + person
                        for cloud in os.listdir(pathPerson):
                            tp = cloud.split('_')[1]
                            if tp == 'N' or tp == 'O':
                                cloud_param = "-cloud {}/{}/{} ".format(
                                    pathPeople, person, cloud)
                                keypoints_param = "-keypoints landmarksManually,{}/{}/{},{}".format(
                                    keypointsPath, person, cloud, region)
                                labels = ",".join(
                                    cloud.split('.')[0].split('_'))
                                output_param = "-output {}:{}".format(
                                    filename, labels)

                                features_param = "-features {}".format(
                                    features)
                                command = "{} {} {} {} {}".format(
                                    exe, cloud_param, keypoints_param,
                                    features_param, output_param)

                                count = count + 1
                                print(
                                    'feito para nuvem ({}) ------> {} - radius_feat: {}, radius_norm {}\nsaved on {}'
                                    .format(cloud, count, radius_feat,
                                            radius_norm, pathOutput))
                                os.system(command)
                    #     break
                    # break


if __name__ == "__main__":
    main()