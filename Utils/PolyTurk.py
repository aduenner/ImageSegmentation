import os
import numpy as np
from PIL import Image, ImageDraw
import csv
import os
import base64
import boto3
import xmltodict
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
import scipy.ndimage.morphology as morphology
import glob
from bs4 import BeautifulSoup
import ast
from datetime import datetime

class Turk(object):
    def __init__(self, user_inputs):
        # MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
        MTURK_PRODUCTION = "https://mturk-requester.us-east-1.amazonaws.com"
        self.mturk = boto3.client('mturk', endpoint_url = MTURK_PRODUCTION)
        self.HITs = None
        self.AssignmentCount = None
        self.Paths = user_inputs
        self.TaskData = None
        self.Overwrite = False
        
        self.getHITs()
        self.getAssignmentCount()
    


    def polygon_to_mask(self,polygons,mask_name):
       
        polygon_count = len(polygons)
        color_dict = {
            'Nucleus': (44,160,44,255),
            'Goblet': (255,127,14,255),
            'Microvilli': (31,110,180,255),
            'Basement': (214,39,40,255)
            }

        img =  Image.new('RGBA', (1024,1024), (255,255,255,0))
        polygon_segments = np.zeros((polygon_count,1))
        for idx,polygon in enumerate(polygons):
            label = polygon['label']
            vertices = polygon['vertices']
            coords = [(v['x'], v['y']) for v in vertices]
            polygon_segments[idx] = len(coords)
            if len(coords)>2:
                ImageDraw.Draw(img).polygon(coords, outline=1, fill=color_dict[label])
    
        if polygon_count>1:
            mask = np.array(img)
            if not os.path.isfile(mask_name) or self.Overwrite:
                img.save(mask_name,"PNG")
        avg_polygon_segments = np.average(polygon_segments)        
    
        return polygon_count, avg_polygon_segments    

    
    def getHITs(self):
        query = self.mturk.list_hits(MaxResults=100)
        hits = query['HITs']
        token = query['NextToken']
        hit_Ids = [h['HITId'] for h in hits]
        count=0;
        while token:
            query = self.mturk.list_hits(NextToken=token, MaxResults=100)
            if 'NextToken' in query.keys():
                token=query['NextToken']
            else:
                token=''
            hits = query['HITs']
            these_hit_ids = [h['HITId'] for h in hits]
            if these_hit_ids:
                hit_Ids = hit_Ids + these_hit_ids
        self.HITs = hit_Ids
        print('Number of HITs: '+str(len(self.HITs)))

    def getAssignmentCount(self):
        num_assignments=0
        for hit in self.HITs:
            results = self.mturk.list_assignments_for_hit(HITId=hit, AssignmentStatuses=['Submitted'])['NumResults']
            num_assignments+=results
        self.AssignmentCount = num_assignments
        print('Reviewable Assignments: '+str(num_assignments))

    def getTaskData(self):    
        task_data=[]
        num_tasks = 0
        for idx,hit in enumerate(self.HITs):
            results = self.mturk.list_assignments_for_hit(HITId=hit, AssignmentStatuses=['Submitted'])
            assignments=[]
            for idy,assignment in enumerate(results['Assignments']):
                HID = assignment['HITId']
                AID = assignment['AssignmentId']
                WID = assignment['WorkerId']
                HIT = self.mturk.get_assignment(AssignmentId=AID)['HIT']
        
                Question = xmltodict.parse(HIT['Question'])
                qhtml = Question['HTMLQuestion']['HTMLContent']
                soup = BeautifulSoup(qhtml)
                image_url = soup.find("crowd-polygon")['src']
                image_name =  image_url.split('/')[-1].strip('.png')
        
                answer = xmltodict.parse(assignment['Answer'])
                answer_value = answer['QuestionFormAnswers']['Answer'][-1]['FreeText']
                polygon = ast.literal_eval(answer_value)
                mask_name = image_name.strip('.png')+'_'+AID[-5:]+'.png'
                mask_path = self.Paths['MASK_PATH'] + mask_name
                polygon_count, avg_poly_segments = self.polygon_to_mask(polygon,mask_path)
                if polygon_count>1:
                    score=10
                else:
                    score=0
            
                task_data.append({
                    "Mask_Name": mask_name,
                    "Image_ID": image_name.strip('SemImage'),
                    "AssignmentID": AID,
                    "Worker": WID,
                    "Answer": polygon,
                    "Regions": polygon_count,
                    "Avg_Poly_Verts": avg_poly_segments,
                    "Submit_Time": assignment['SubmitTime'].strftime("%m/%d/%Y, %H:%M:%S"),
                    "Duration": (assignment['SubmitTime']-assignment['AcceptTime']).seconds,
                    "Image_Path": self.Paths['IMG_PATH'] + image_name + '.png',
                    "Mask_Path": self.Paths['MASK_PATH'] + mask_name,
                    "Approved": False,
                    "Score": score
                })
        self.TaskData = task_data
        return task_data