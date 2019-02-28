import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

def change():
    path = "/home/jkwang/learn_sumo/straight"
    os.chdir(path)
    cmd = '/usr/share/sumo/tools/randomTrips.py -n straight.net.xml -r straight.rou.xml -p 0.5 -e 300 --trip-attributes=\"departLane=\\"best\\" departSpeed=\\"max\\" color=\\"0,1,1\\" departPos=\\"random\\"\"'
    print(cmd)
    os.system(cmd)
    tree = read_xml()
    agent = gen_element()
    root = tree.getroot()
    #root[100] = agent
    root[100].attrib['id'] = 'agent'
    root[100].attrib['color'] = '1,0,0'
    root[100][0].attrib['edges'] = 'gneE0 -gneE1'

    write_xml(tree,'/home/jkwang/learn_sumo/straight/straight.rou.xml')

def write_xml(tree,out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def read_xml():
    tree = ET.parse('/home/jkwang/learn_sumo/straight/straight.rou.xml')
    return tree

def gen_element():
    vehicle = Element('vehicle',{"id":'agent',"depart":'100',"departPos":'random','departSpeed':'random','color':"1,0,0"})
    route = Element('route',{'edges':'gneE0 -gneE1'})
    vehicle.append(route)

    return vehicle
