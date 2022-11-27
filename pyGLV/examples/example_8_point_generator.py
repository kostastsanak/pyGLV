

from statistics import mode
from telnetlib import GA
from turtle import position, width
# import unittest

import numpy as np
# from sympy import true

import pyECSS.utilities as util
from pyECSS.Entity import Entity
from pyECSS.Component import BasicTransform, Camera, RenderMesh
from pyECSS.System import System, TransformSystem, CameraSystem, RenderSystem
from pyparsing import line
from stack_data import Line
from pyGLV.GL.Scene import Scene
from pyECSS.ECSSManager import ECSSManager
from pyGLV.GUI.Viewer import SDL2Window, ImGUIDecorator, RenderGLStateSystem, ImGUIecssDecorator

from pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from pyGLV.GL.VertexArray import VertexArray
from OpenGL.GL import GL_LINES

import OpenGL.GL as gl
import keyboard as key
import time
import random
from pyECSS.Event import Event, EventManager
import pandas as pd

import sys

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import itertools
import os
import imgui

"""
Common setup for all unit tests

Scenegraph for unit tests:

root
    |------------------------------------------------------|
    |                                                      |-LineNode
    |                                                      |-------------|-----------|-------------|          
    |                                                      trans5        mesh5      shaderdec5      vArray5
    |------------------------------------------------------|
    |                                                      |-SplineNode
    |                                                      |-------------|-----------|-------------|          
    |                                                      trans6        mesh6      shaderdec6      vArray6

    |------------------------------------------------------|
    |                                                      |-SuperFunction
    |                                                      |-------------|-----------|-------------|          
    |                                                      trans8        mesh8      shaderdec8      vArray8
    |------------------------------------------------------|
    |                                                      |-TriangleNode
    |                                                      |-------------|-----------|-------------|          
    |                                                       trans7        mesh7      shaderdec7      vArray7
    |
    |------------------------------------------------------|-Histogram
    |                                                      |-------------|-----------|-------------|          
    |---------------------------|                          trans7        mesh7      shaderdec7      vArray7
    entityCam1,                 PointsNode,      
    |-------|                    |--------------|----------|--------------|           
    trans1, entityCam2           trans4,        mesh4,     shaderDec4     vArray4
            |                               
            ortho, trans2                   

"""
#get random x,y,z planes
if len(sys.argv) == 2:#receive data from a txt via command line
    with open(sys.argv[1], 'r') as f:
        next(f)#skip header
        if(f):
            pointListfromCSV = [tuple(map(float, i.split(','))) for i in f]
else:#receive csv from path
    reader = pd.read_csv("pyGLV/examples/example_materials/PointCoordinates.csv")
    pointListfromCSV = [tuple(x) for x in reader.values]

df = pd.DataFrame(pointListfromCSV, index=None)
cols = len(df.axes[1]) -2
xPlane = random.randint(0, cols)
yPlane = random.randint(0, cols)
#zPlane = random.randint(0, cols)

while((yPlane) == xPlane):# having x and y plane from the same csv collumn is ugly!
    yPlane = random.randint(0, cols)
#while((zPlane) == xPlane or (zPlane) == yPlane):# having x and y plane from the same csv collumn is ugly!
#   zPlane = random.randint(0, cols)
zPlane = cols+1

scene = Scene()    
rootEntity = scene.world.createEntity(Entity(name="RooT"))
entityCam1 = scene.world.createEntity(Entity(name="entityCam1"))
scene.world.addEntityChild(rootEntity, entityCam1)
trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()))

entityCam2 = scene.world.createEntity(Entity(name="entityCam2"))
scene.world.addEntityChild(entityCam1, entityCam2)
trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()))
orthoCam = scene.world.addComponent(entityCam2, Camera(util.ortho(-100.0, 100.0, -100.0, 100.0, 2.0, 100.0), "orthoCam","Camera","500"))

PointsNode = scene.world.createEntity(Entity("PointsNode"))
scene.world.addEntityChild(scene.world.root, PointsNode)
trans4 = BasicTransform(name="trans4", trs=util.identity())
scene.world.addComponent(PointsNode, trans4)

LinesNode = scene.world.createEntity(Entity("LinesNode"))
scene.world.addEntityChild(rootEntity, LinesNode)
trans5 = BasicTransform(name="trans5", trs=util.identity())
scene.world.addComponent(LinesNode, trans5)

SplineNode = scene.world.createEntity(Entity("SplineNode"))
scene.world.addEntityChild(rootEntity, SplineNode)
trans6 = BasicTransform(name="trans6", trs=util.identity())
scene.world.addComponent(SplineNode, trans6)

Area3D = scene.world.createEntity(Entity("TriangleNode"))
scene.world.addEntityChild(rootEntity, Area3D)
trans7 = BasicTransform(name="trans7", trs=util.identity())
scene.world.addComponent(Area3D, trans7)

Area2D = scene.world.createEntity(Entity("TriangleNode"))
scene.world.addEntityChild(rootEntity, Area2D)
trans11 = BasicTransform(name="trans11", trs=util.identity())
scene.world.addComponent(Area2D, trans11)

SuperFunction = scene.world.createEntity(Entity("SuperFunction"))
scene.world.addEntityChild(rootEntity, SuperFunction)
trans8 = BasicTransform(name="trans8", trs=util.identity())
scene.world.addComponent(SuperFunction, trans8)

Histogram2D = scene.world.createEntity(Entity("Histogram"))
scene.world.addEntityChild(rootEntity, Histogram2D)
trans9 = BasicTransform(name="trans9", trs=util.identity())
scene.world.addComponent(Histogram2D, trans9)

Histogram3D = scene.world.createEntity(Entity("Histogram"))
scene.world.addEntityChild(rootEntity, Histogram3D)
trans10 = BasicTransform(name="trans10", trs=util.identity())
scene.world.addComponent(Histogram3D, trans10)

axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

#global point size integer
keys = []
values = []

CSVxyValues= xPlane,yPlane
CSVButtonclicked = 0
def CSV_GUI():
    global xPlane
    global yPlane
    global CSVxyValues
    global CSVButtonclicked

    imgui.begin("- Program - Select CSV Dimension Values")
    if(CSVButtonclicked):
        imgui.same_line() 

        imgui.text("X: %d, Y: %d" % (CSVxyValues[0],CSVxyValues[1]))
        xPlane = CSVxyValues[0]
        yPlane = CSVxyValues[1]
        CSVButtonclicked = imgui.button("change")
        CSVButtonclicked = (CSVButtonclicked - 1) * (CSVButtonclicked - 1)
    else:
        imgui.text("Choose Dimension from CSV file for X,Y ")
        changed, CSVxyValues = imgui.input_int2('', *CSVxyValues) 
        l=0
        for l in range(len(CSVxyValues)):
            if(CSVxyValues[l] > cols):
                CSVxyValues[l] = cols
            elif(CSVxyValues[l] < 0):
                CSVxyValues[l] = 0

        #imgui.same_line() 
        imgui.text("X: %d, Y: %d" % (CSVxyValues[0],CSVxyValues[1]))
        CSVButtonclicked = imgui.button("Save")
    #implementation for csv import
    if imgui.is_item_hovered():
        imgui.set_tooltip("Please save the changes or they won't be passed")
    imgui.end()

FuncValues= 0.01,0.3,1.,1.
FuncButtonclicked = 0
f_x_y = 'x**2+x*4'
Func_Button =0
superfuncchild =0
toggleSuperFunc = True
def Func_GUI():
    global FuncValues
    global FuncButtonclicked
    global f_x_y
    global Func_Button
    global superfuncchild
    global toggleSuperFunc
    imgui.begin("Give Function X,Y Values")

    #implementation for a,b,c,d points for X,Y functions
    imgui.text("Give a and b values for X() and c and d for Y() functions")
    changed, FuncValues = imgui.input_float4('', *FuncValues) 
    #imgui.same_line() 
    imgui.text("a: %.1f, b: %.1f, c: %.1f, d: %.1f" % (FuncValues[0],FuncValues[1],FuncValues[2],FuncValues[3]))

    changed, f_x_y = imgui.input_text('Amount:',f_x_y,256)
    Func_Button = imgui.button("Print Function")
    if(Func_Button):
        if superfuncchild != 0:
            toggleSuperFunc = not toggleSuperFunc
        else:
            l=0
            x = np.linspace(FuncValues[0],FuncValues[1],100) 
            y = np.linspace(FuncValues[2],FuncValues[3],100) 
            z= f_Z(x,y)
            while (l < len(x)-1):
                superfuncchild+=1
                l+=1
                DynamicVariable = "SuperFunction" + str(superfuncchild)
                point1 =  x[l], y[l], z[l] , 1 
                point2 =  x[l-1], y[l-1], z[l-1] , 1
                vars()[DynamicVariable]: GameObjectEntity = LineSpawn(DynamicVariable,point2,point1, 1, 1, 0)
                scene.world.addEntityChild(SuperFunction, vars()[DynamicVariable])
            scene.world.traverse_visit(initUpdate, scene.world.root)
    
    imgui.end()


Area_Diagram_Button3D = 0
Area_Diagram_Button2D = 0

SplineList = []
togglePlatformSwitch3D = False
togglePlatformSwitch2D = False

trianglechild3D =0
trianglechild2D =0

def Area_Chart():
    global Area_Diagram_Button3D
    global Area_Diagram_Button2D

    global SplineList
    global keys
    global values

    global togglePlatformSwitch3D
    global togglePlatformSwitch2D
    global trianglechild3D
    global trianglechild2D
    imgui.begin("- Calculate Area 3D -")
    imgui.text("Calculate Area 3D ")
    Area_Diagram_Button3D = imgui.button("3D")
    imgui.same_line() 
    Area_Diagram_Button2D = imgui.button("2D")

    if(Area_Diagram_Button3D):
        SplineList.clear()
        for i in range(len(keys)):
            if(values[i]):
                arr = np.array(values[i])
                xPointToSpline = arr[:,xPlane]
                yPointToSpline = arr[:,yPlane]
                
                f = CubicSpline(xPointToSpline,yPointToSpline, bc_type='natural')
                x_new = np.linspace(0.2, 3, len(values[i])*6)
                SplineList.append(f)#pass the spline functions to a list for later use
        lengthoflist = 0
        #get random rgb color for the platform
        r = random.uniform(0, 1.0)
        g = random.uniform(0, 1.0)
        b = random.uniform(0, 1.0)
        while(lengthoflist < len(SplineList) -1):
            lengthoflist += 1
            spline1 = SplineList[lengthoflist-1]
            spline2 = SplineList[lengthoflist]
            x_new = np.linspace(0.2, 3, len(values[i])*6)

            y_spline1 = spline1(x_new)
            y_spline2 = spline2(x_new)
            z_spline1 = keys[lengthoflist-1]
            z_spline2 = keys[lengthoflist]
            l =0
            if(trianglechild3D==0):#SplineList and trianglechild3D==0
                while l < len(x_new) -1 :
                    l += 1
                    #first triangle
                    trianglechild3D += 1
                    DynamicVariable = "Triangle" + str(trianglechild3D)
                    point1 = x_new[l-1], y_spline1[l-1], z_spline1, 1
                    point2 = x_new[l], y_spline1[l], z_spline1, 1
                    point3 = x_new[l-1], y_spline2[l-1], z_spline2, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3,r ,g, b)
                    scene.world.addEntityChild(Area3D, vars()[DynamicVariable])
                    #second triangle
                    trianglechild3D += 1
                    DynamicVariable = "Triangle" + str(trianglechild3D)
                    point1 = x_new[l-1], y_spline2[l-1], z_spline2, 1
                    point2 = x_new[l], y_spline2[l], z_spline2, 1
                    point3 = x_new[l], y_spline1[l], z_spline1, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3,r ,g, b)
                    scene.world.addEntityChild(Area3D, vars()[DynamicVariable])
                scene.world.traverse_visit(initUpdate, scene.world.root)
        else:
            togglePlatformSwitch3D = not togglePlatformSwitch3D
    elif(Area_Diagram_Button2D):
        SplineList.clear()
        for i in range(len(keys)):
            if(values[i]):
                arr = np.array(values[i])
                xPointToSpline = arr[:,xPlane]
                yPointToSpline = arr[:,yPlane]
                
                f = CubicSpline(xPointToSpline,yPointToSpline, bc_type='natural')
                x_new = np.linspace(0.2, 3, len(values[i])*6)
                SplineList.append(f)#pass the spline functions to a list for later use
        lengthoflist = 0
        #get random rgb color for the platform
        r = random.uniform(0, 1.0)
        g = random.uniform(0, 1.0)
        b = random.uniform(0, 1.0)
        while(lengthoflist < len(SplineList) -1):
            lengthoflist += 1
            spline1 = SplineList[lengthoflist-1]
            spline2 = SplineList[lengthoflist]
            x_new = np.linspace(0.2, 3, len(values[i])*6)

            y_spline1 = spline1(x_new)
            y_spline2 = spline2(x_new)
            z_spline1 = keys[lengthoflist-1]
            z_spline2 = keys[lengthoflist]
            l =0
            if(trianglechild2D==0):#SplineList and trianglechild2D==0
                while l < len(x_new) -1 :
                    l += 1
                    #first triangle
                    trianglechild2D += 1
                    DynamicVariable = "Triangle" + str(trianglechild2D)
                    point1 = x_new[l-1], y_spline1[l-1], 0, 1
                    point2 = x_new[l], y_spline1[l], 0, 1
                    point3 = x_new[l-1], y_spline2[l-1], 0, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3,r ,g, b)
                    scene.world.addEntityChild(Area2D, vars()[DynamicVariable])
                    #second triangle
                    trianglechild2D += 1
                    DynamicVariable = "Triangle" + str(trianglechild2D)
                    point1 = x_new[l-1], y_spline2[l-1], 0, 1
                    point2 = x_new[l], y_spline2[l], 0, 1
                    point3 = x_new[l], y_spline1[l], 0, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3,r ,g, b)
                    scene.world.addEntityChild(Area2D, vars()[DynamicVariable])
                scene.world.traverse_visit(initUpdate, scene.world.root)
        else:
            togglePlatformSwitch2D = not togglePlatformSwitch2D
    #implementation for csv import
    if imgui.is_item_hovered():
        imgui.set_tooltip("Please save the changes or they won't be passed")
    imgui.end()


Scatterplot_Button = 0
Scatterplot_Button2D =0
Scatterplot_Button3D =0
toggle_scatterplot_Button = 0
toggle_scatterplot = True
pointchild = 0
PointSize = 5    
PointsColor = 0., 1., 1., 1
def ScatterPlot_Chart():
    global pointchild
    global Scatterplot_Button3D
    global Scatterplot_Button2D
    global toggle_scatterplot_Button
    global toggle_scatterplot

    imgui.begin("- Calculate Scatterplot -")
    imgui.text("Scatterplot ")
    Scatterplot_Button2D = imgui.button("2D Scatterplot")
    imgui.same_line() 
    Scatterplot_Button3D = imgui.button("3D Scatterplot")

    if (Scatterplot_Button2D or Scatterplot_Button3D):
        if (pointchild == 0 ):
            for i in range(len(pointListfromCSV)):
                pointchild += 1
                DynamicVariable = "Point" + str(pointchild)
                vars()[DynamicVariable]: GameObjectEntity_Point = PointSpawn(DynamicVariable,0,1,1)
                scene.world.addEntityChild(PointsNode, vars()[DynamicVariable]) 
                PointsNode.getChild(pointchild).trans.l2cam =  util.translate(pointListfromCSV[i][xPlane], pointListfromCSV[i][yPlane], pointListfromCSV[i][zPlane])
            scene.world.traverse_visit(initUpdate, scene.world.root)

        pointchild = 0
        for i in range(len(pointListfromCSV)):
            pointchild += 1
            if Scatterplot_Button3D:
                PointsNode.getChild(pointchild).trans.l2cam =  util.translate(pointListfromCSV[i][xPlane], pointListfromCSV[i][yPlane], pointListfromCSV[i][zPlane])
            elif Scatterplot_Button2D:
                PointsNode.getChild(pointchild).trans.l2cam =  util.translate(pointListfromCSV[i][xPlane], pointListfromCSV[i][yPlane], 0)
        
        time.sleep(0.15)

    if imgui.is_item_hovered():
        imgui.set_tooltip("Please save the changes or they won't be passed")

    #size, color of scaterplot(points)
    global PointSize
    global PointsColor
    changed, PointSize = imgui.drag_float("Point Size", PointSize, 0.02, 0.1, 40, "%.1f")
    if (changed):
        gl.glPointSize(PointSize)
    imgui.text("PointSize: %s" % (PointSize))
    imgui.text("")
    changed, PointsColor = imgui.color_edit3("Color", *PointsColor)

    toggle_scatterplot_Button = imgui.button("Toggle Scaterplot On/Off")
    if (toggle_scatterplot_Button):
        toggle_scatterplot = not toggle_scatterplot
        
    imgui.end()


histogramchild2D = 0
histogramchild3D = 0
Histogram_Button2D = 0
Histogram_Button3D = 0
detailedHistogram = 10
toggle2DHistogram = True
toggle3DHistogram = True
def Histogram_Chart():
    global histogramchild2D
    global histogramchild3D

    global Histogram_Button2D
    global Histogram_Button3D
    global detailedHistogram
    global toggle2DHistogram
    global toggle3DHistogram
    imgui.begin("- Calculate Histogram -")
    imgui.text("Histogram ")
    Histogram_Button2D = imgui.button("2D Histogram")
    imgui.same_line() 
    Histogram_Button3D = imgui.button("3D Histogram")
    changed, detailedHistogram = imgui.input_int('', detailedHistogram) 
    bins = np.linspace(0, 3, detailedHistogram)
 
    if (Histogram_Button2D):
        arrayfromtupple = np.asarray(pointListfromCSV)
        HistogramY,HistogramX = np.histogram(arrayfromtupple[:,xPlane], bins )#= [0, 0.4, 0.8, 1.2, 1.6,2.0, 2.4, 2.8]
        i=0
        if(histogramchild2D == 0):
            while i < len(HistogramX) -1:
                r = random.uniform(0, 1.0)
                g = random.uniform(0, 1.0)
                b = random.uniform(0, 1.0)
                i+=1
                histogramchild2D+=1
                DynamicVariable = "Cube" + str(histogramchild2D)

                point1 = HistogramX[i-1], 0, 0
                point2 = HistogramX[i-1], HistogramY[i-1], 0
                point3 = HistogramX[i], HistogramY[i-1], 0
                point4 = HistogramX[i], 0, 0              
                point5 = HistogramX[i-1], 0, 0
                point6 = HistogramX[i-1], HistogramY[i-1], 0
                point7 = HistogramX[i], HistogramY[i-1], 0
                point8 = HistogramX[i], 0, 0
                
                vars()[DynamicVariable]: GameObjectEntity = CubeSpawn(DynamicVariable,point1,point2,point3,point4,point5,point6,point7,point8,r,g,b)
                scene.world.addEntityChild(Histogram2D, vars()[DynamicVariable])
            scene.world.traverse_visit(initUpdate, scene.world.root)
        else:
            toggle2DHistogram = not toggle2DHistogram
    elif(Histogram_Button3D):
        if(histogramchild3D == 0):
            for x in range(len(keys)):
                if(values[x]):
                    arr = np.array(values[x])
                HistogramY,HistogramX = np.histogram(arr[:,xPlane], bins )#= [0, 0.4, 0.8, 1.2, 1.6,2.0, 2.4, 2.8]
                i=0
                while i < len(HistogramX) -1:
                    r = random.uniform(0, 1.0)
                    g = random.uniform(0, 1.0)
                    b = random.uniform(0, 1.0)
                    i+=1
                    histogramchild3D+=1
                    DynamicVariable = "Cube" + str(histogramchild3D)

                    point1 = HistogramX[i-1], 0, keys[x]+1
                    point2 = HistogramX[i-1], HistogramY[i-1], keys[x]+1
                    point3 = HistogramX[i], HistogramY[i-1], keys[x]+1
                    point4 = HistogramX[i], 0, keys[x]+1              
                    point5 = HistogramX[i-1], 0, keys[x]
                    point6 = HistogramX[i-1], HistogramY[i-1], keys[x]
                    point7 = HistogramX[i], HistogramY[i-1], keys[x]
                    point8 = HistogramX[i], 0, keys[x]
                    
                    vars()[DynamicVariable]: GameObjectEntity = CubeSpawn(DynamicVariable,point1,point2,point3,point4,point5,point6,point7,point8,r,g,b)
                    scene.world.addEntityChild(Histogram3D, vars()[DynamicVariable])
            scene.world.traverse_visit(initUpdate, scene.world.root)
        else:
            toggle3DHistogram = not toggle3DHistogram
        
    imgui.end()


def displayGUI():
    """
        displays ImGui
    """
    #global value
    Func_GUI()
    CSV_GUI()
    Area_Chart()
    ScatterPlot_Chart()
    Histogram_Chart()


def f_Z (x,y):
    global f_x_y
    d= {}
    d['x'] = x
    d['y'] = y
    z = eval(f_x_y,d)
    return z


COLOR_FRAG = """
    #version 410

    in vec4 color;
    out vec4 outputColor;

    void main()
    {
        outputColor = color;
        //outputColor = vec4(0.1, 0.1, 0.1, 1);
    }
"""
COLOR_VERT_MVP = """
    #version 410

    layout (location=0) in vec4 vPosition;
    layout (location=1) in vec4 vColor;

    out     vec4 color;
    uniform mat4 modelViewProj;
    uniform vec4 extColor;

    void main()
    {

        gl_Position = modelViewProj * vPosition;
        color = extColor;
    }
"""

class IndexedConverter():
    
    # Assumes triangulated buffers. Produces indexed results that support
    # normals as well.
    def Convert(self, vertices, colors, indices, produceNormals=True):

        iVertices = [];
        iColors = [];
        iNormals = [];
        iIndices = [];
        for i in range(0, len(indices), 3):
            iVertices.append(vertices[indices[i]]);
            iVertices.append(vertices[indices[i + 1]]);
            iVertices.append(vertices[indices[i + 2]]);
            iColors.append(colors[indices[i]]);
            iColors.append(colors[indices[i + 1]]);
            iColors.append(colors[indices[i + 2]]);
            

            iIndices.append(i);
            iIndices.append(i + 1);
            iIndices.append(i + 2);

        if produceNormals:
            for i in range(0, len(indices), 3):
                iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));

        iVertices = np.array( iVertices, dtype=np.float32 )
        iColors   = np.array( iColors,   dtype=np.float32 )
        iIndices  = np.array( iIndices,  dtype=np.uint32  )

        iNormals  = np.array( iNormals,  dtype=np.float32 )

        return iVertices, iColors, iIndices, iNormals;
class GameObjectEntity_Point(Entity):
    def __init__(self, name=None, type=None, id=None, primitiveID = gl.GL_LINES) -> None:
        super().__init__(name, type, id)

        # Gameobject basic properties
        self._color          = [1, 0.5, 0.2, 1.0] # this will be used as a uniform var
        # Create basic components of a primitive object
        self.trans          = BasicTransform(name="trans", trs=util.identity())
        self.mesh           = RenderMesh(name="mesh")
        # self.shaderDec      = ShaderGLDecorator(Shader(vertex_source=Shader.VERT_PHONG_MVP, fragment_source=Shader.FRAG_PHONG))
        self.shaderDec      = ShaderGLDecorator(Shader(vertex_source= COLOR_VERT_MVP, fragment_source=COLOR_FRAG))
        self.vArray         = VertexArray(primitive= primitiveID)
        # Add components to entity
        scene = Scene()
        scene.world.createEntity(self)
        scene.world.addComponent(self, self.trans)
        scene.world.addComponent(self, self.mesh)
        scene.world.addComponent(self, self.shaderDec)
        scene.world.addComponent(self, self.vArray)
        

    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, colorArray):
        self._color = colorArray

    def drawSelfGui(self, imgui):
        changed, value = imgui.color_edit3("Color", self.color[0], self.color[1], self.color[2])
        self.color = [value[0], value[1], value[2], 1.0]

    def SetVertexAttributes(self, vertex, color, index, normals = None):
        self.mesh.vertex_attributes.append(vertex)
        self.mesh.vertex_attributes.append(color)
        self.mesh.vertex_index.append(index)
        if normals is not None:
            self.mesh.vertex_attributes.append(normals)

class GameObjectEntity(Entity):

    def __init__(self, name=None, type=None, id=None, primitiveID = gl.GL_LINES) -> None:
        super().__init__(name, type, id)

        # Gameobject basic properties
        self._color          = [1, 0.5, 0.2, 1.0] # this will be used as a uniform var
        # Create basic components of a primitive object
        self.trans          = BasicTransform(name="trans", trs=util.identity())
        self.mesh           = RenderMesh(name="mesh")
        # self.shaderDec      = ShaderGLDecorator(Shader(vertex_source=Shader.VERT_PHONG_MVP, fragment_source=Shader.FRAG_PHONG))
        self.shaderDec      = ShaderGLDecorator(Shader(vertex_source= Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG))
        self.vArray         = VertexArray(primitive= primitiveID)
        # Add components to entity
        scene = Scene()
        scene.world.createEntity(self)
        scene.world.addComponent(self, self.trans)
        scene.world.addComponent(self, self.mesh)
        scene.world.addComponent(self, self.shaderDec)
        scene.world.addComponent(self, self.vArray)
        

    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, colorArray):
        self._color = colorArray

    def drawSelfGui(self, imgui):
        changed, value = imgui.color_edit3("Color", self.color[0], self.color[1], self.color[2])
        self.color = [value[0], value[1], value[2], 1.0]

    def SetVertexAttributes(self, vertex, color, index, normals = None):
        self.mesh.vertex_attributes.append(vertex)
        self.mesh.vertex_attributes.append(color)
        self.mesh.vertex_index.append(index)
        if normals is not None:
            self.mesh.vertex_attributes.append(normals)

def CubeSpawn(cubename = "Cube",p1=[-0.5, -0.5, 0.5, 1.0],p2 = [-0.5, 0.5, 0.5, 1.0],p3 = [0.5, 0.5, 0.5, 1.0]
, p4 = [0.5, -0.5, 0.5, 1.0], p5=[-0.5, -0.5, -0.5, 1.0], p6=[-0.5, 0.5, -0.5, 1.0], p7=[0.5, 0.5, -0.5, 1.0], p8=[0.5, -0.5, -0.5, 1.0], r=0.55,g=0.55,b=0.55): 
    cube = GameObjectEntity(cubename, primitiveID=gl.GL_TRIANGLES)
    vertices = [
        p1,p2,p3,p4,p5,p6,p7,p8        
    ]
    colors = [
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0]                
    ]
    
    #index arrays for above vertex Arrays
    indices = np.array(
        (
            1,0,3, 1,3,2, 
            2,3,7, 2,7,6,
            3,0,4, 3,4,7,
            6,5,1, 6,1,2,
            4,5,6, 4,6,7,
            5,4,0, 5,0,1
        ),
        dtype=np.uint32
    ) #rhombus out of two triangles

    #vertices, colors, indices, normals = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True);
    cube.SetVertexAttributes(vertices, colors, indices, None)
    
    return cube


def TriangleSpawn(trianglename = "Triangle",p1=[0,0,0,1],p2 = [0.4,0.4,0,1],p3 = [0.8,0.0,0,1], r=0.55,g=0.55,b=0.55):
    triangle = GameObjectEntity(trianglename,primitiveID=gl.GL_TRIANGLES)
    vertices = [
        p1,p2,p3
    ]
    colors = [
        [r, g, b, 1.0],
        [r, g, b, 1.0],
        [r, g, b, 1.0]
    ]
    
    indices = np.array(
        (
            1,0,2
        ),
        dtype=np.uint32
    ) 
    #vertices, colors, indices, normals = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True)

    #triangle.SetVertexAttributes(vertices, colors, indices, normals)
    triangle.SetVertexAttributes(vertices, colors, indices, None)

    
    return triangle

def LineSpawn(linename = "Line",p1=[0,0,0,1],p2 = [0.4,0.4,0,1], r=0.7,g=0.7,b=0.7):
    line = GameObjectEntity(linename,primitiveID=gl.GL_LINES)
    vertices = [
        p1,p2
    ]
    colors = [
        [r, g, b, 1.0],
        [r, g, b, 1.0] 
    ]
    
    indices = np.array(
        (
            0,1,3
        ),
        dtype=np.uint32
    ) 

    #vertices, colors, indices, none = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True)
    line.SetVertexAttributes(vertices, colors, indices, None)
    
    return line

def PointSpawn(pointname = "Point",r=0.,g=1.,b=1.): 
    point = GameObjectEntity_Point(pointname,primitiveID=gl.GL_POINTS)
    vertices = [
        
        [0, 0, 0, 1.0]
    ]
    colors = [
        
        [r, g, b, 1.0]                    
    ]
    indices = np.array(
        (
            0
        ),
        dtype=np.uint32
    ) 

    #vertices, colors, indices, normals = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True)
    point.SetVertexAttributes(vertices, colors, indices, None)
    
    return point

def main (imguiFlag = False):
    #Colored Axes
    vertexAxes = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ],dtype=np.float32) 
    colorAxes = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float32)

    #index arrays for above vertex Arrays
    #index = np.array((0,1,2), np.uint32) #simple triangle
    indexAxes = np.array((0,1,2,3,4,5), np.uint32) #3 simple colored Axes as R,G,B lines

    # Systems
    transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
    camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
    renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
    initUpdate = scene.world.createSystem(InitGLShaderSystem())

    """
    test_renderPointGenaratorEVENT
    """
    # Generate terrain
    from pyGLV.GL.terrain import generateTerrain
    vertexTerrain, indexTerrain, colorTerrain= generateTerrain(size=4,N=20)
    # Add terrain
    terrain = scene.world.createEntity(Entity(name="terrain"))
    scene.world.addEntityChild(rootEntity, terrain)
    terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.identity()))
    terrain_mesh = scene.world.addComponent(terrain, RenderMesh(name="terrain_mesh"))
    terrain_mesh.vertex_attributes.append(vertexTerrain) 
    terrain_mesh.vertex_attributes.append(colorTerrain)
    terrain_mesh.vertex_index.append(indexTerrain)
    terrain_vArray = scene.world.addComponent(terrain, VertexArray(primitive=GL_LINES))
    terrain_shader = scene.world.addComponent(terrain, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))
    # terrain_shader.setUniformVariable(key='modelViewProj', value=mvpMat, mat4=True)
   
    ## ADD AXES ##

    axes = scene.world.createEntity(Entity(name="axes"))
    scene.world.addEntityChild(rootEntity, axes)
    axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
    axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
    axes_mesh.vertex_attributes.append(vertexAxes) 
    axes_mesh.vertex_attributes.append(colorAxes)
    axes_mesh.vertex_index.append(indexAxes)
    axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES))

    # OR
    axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))
    
    running = True
    # MAIN RENDERING LOOP
    scene.init(imgui=True, windowWidth = 1024, windowHeight = 768, windowTitle = "pyglGA test_renderAxesTerrainEVENT")#, customImGUIdecorator = ImGUIecssDecorator
    imGUIecss = scene.gContext

    # pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
    # needs an active GL context
    
    # vArrayAxes.primitive = gl.GL_LINES
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glDisable(gl.GL_CULL_FACE)

    # gl.glDepthMask(gl.GL_FALSE)  
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)


    ################### EVENT MANAGER ###################
    scene.world.traverse_visit(initUpdate, scene.world.root)

    eManager = scene.world.eventManager
    gWindow = scene.renderWindow
    gGUI = scene.gContext

    renderGLEventActuator = RenderGLStateSystem()


    #updateTRS = Event(name="OnUpdateTRS", id=100, value=None) #lines 255-258 contains the Scenegraph in a GUI as is it has issues.. To be fixed
    #updateBackground = Event(name="OnUpdateBackground", id=200, value=None)
    #eManager._events[updateTRS.name] = updateTRS
    #eManager._events[updateBackground.name] = updateBackground


    #eManager._subscribers[updateTRS.name] = gGUI
    #eManager._subscribers[updateBackground.name] = gGUI

    eManager._subscribers['OnUpdateWireframe'] = gWindow
    eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
    eManager._subscribers['OnUpdateCamera'] = gWindow 
    eManager._actuators['OnUpdateCamera'] = renderGLEventActuator
    
    # Add RenderWindow to the EventManager publishers
    # eManager._publishers[updateBackground.name] = gGUI
    #eManager._publishers[updateBackground.name] = gGUI ## added


    eye = util.vec(1.2, 4.34, 6.1)
    target = util.vec(0.0, 0.0, 0.0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.lookat(eye, target, up)
    
    projMat = util.perspective(50.0, 1.0, 1.0, 20.0) ## Setting the camera as perspective 

    gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

    model_terrain_axes = util.translate(0.0,0.0,0.0)
    
    
    #LISTS
    LinesList = [] 

    #LISTS
    toggleSwitch = True
    toggleLineSwitch = True
    toggleLineZSwitch = True

    #MOST USEFUL GLOBALS
    linechild = 0
    #ALWAYS CHANGING GLOBALS
    linechildZ = 0#used for z=0 lines
    linechildPlane = 0#used for lines that interconnect z=0 with z=1
    ChildMeadian = 0#used for median point
    global pointListfromCSV
    global xPlane
    global yPlane
    global zPlane
    global PointSize
    global values
    global keys

    global  pointchild
    global toggle_scatterplot

    global SplineList

    global trianglechild3D
    global togglePlatformSwitch3D

    global superfuncchild
    global toggleSuperFunc

    global histogramchild2D
    global histogramchild3D
    global toggle2DHistogram
    global toggle3DHistogram


    #we split our csv based on common Z plane values and pass it to 2 lists for later use
    for Zplanekey, value in itertools.groupby(pointListfromCSV, lambda x: x[zPlane]):
        keys.append(Zplanekey)
        values.append(list(value))
    for i in range(len(keys)):
        values[i].sort(key = lambda row: (row[xPlane]),reverse=False)
    
    gl.glPointSize(PointSize)
    #Displays all nodes created
    def Display():
        i=1
        #print points
        while i<=pointchild:
            if(toggle_scatterplot):
                PointsNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @ PointsNode.getChild(i).trans.l2cam, mat4=True)
                PointsNode.getChild(i).shaderDec.setUniformVariable(key='extColor', value=[PointsColor[0], PointsColor[1], PointsColor[2], 1.0], float4=True) #its porpuse is to change color on my_color vertex by setting the uniform                  
            else:
                PointsNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i +=1
        i = 1
        #print Lines
        while i <= linechild:
            if( linechildPlane + i > linechild):#for lines intertwine z plane
                if(toggleLineZSwitch):
                    LinesNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @LinesNode.getChild(i).trans.l2cam, mat4=True)
                else:
                    LinesNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            else:#for lines inside each z plane
                if(toggleLineSwitch):
                    LinesNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @LinesNode.getChild(i).trans.l2cam, mat4=True)
                    #LinesNode.getChild(i-1).shaderDec.setUniformVariable(key='my_color', value=[1, 0, 0] , mat4=True)                   

                else:
                    LinesNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)

            i+=1
        i=1
        #print Area3D
        while i <= trianglechild3D:
            if togglePlatformSwitch3D:
                Area3D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @Area3D.getChild(i).trans.l2cam, mat4=True)
            else:
                Area3D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
        i=1
        #print Area2D
        while i <= trianglechild2D:
            if togglePlatformSwitch2D:
                Area2D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @Area2D.getChild(i).trans.l2cam, mat4=True)
            else:
                Area2D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
        i=1
        #print SuperFunction
        while i <= superfuncchild:
            if toggleSuperFunc:
                SuperFunction.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @SuperFunction.getChild(i).trans.l2cam, mat4=True)
            else:
                SuperFunction.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
        i=1
        #print Histogram2D
        while i<= histogramchild2D:
            if(toggle2DHistogram):
                Histogram2D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @Histogram2D.getChild(i).trans.l2cam, mat4=True)
            else:
                Histogram2D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
        i=1
        #print Histogram3D
        while i<= histogramchild3D:
            if(toggle3DHistogram):
                Histogram3D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @Histogram3D.getChild(i).trans.l2cam, mat4=True)
            else:
                Histogram3D.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
        i=1
        scene.render_post()
    while running:
        running = scene.render(running)
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        view =  gWindow._myCamera # updates view via the imgui
        mvp_point = projMat @ view 
        mvp_terrain_axes = projMat @ view @ model_terrain_axes
        displayGUI()
        #Toggle mechanism
        if (key.is_pressed("t")):#create points 
            toggleSwitch = not toggleSwitch
            time.sleep(0.15)
        if (toggleSwitch):
            axes_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4= True)
            terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4= True)
        else:
            axes_shader.setUniformVariable(key='modelViewProj', value=None, mat4= True)
            terrain_shader.setUniformVariable(key='modelViewProj', value=None, mat4= True)
        #Print Points Mechanism
        if (key.is_pressed("r")):#create points 
            time.sleep(0.15)
        #Print Median Point Mechanism
        elif (key.is_pressed("m")):
            DynamicVariable = "Point" + str(ChildMeadian)
            MOx = 0
            MOy = 0
            while i<=pointchild:
                if(PointsNode.getChild(i).trans.l2cam[2][3] == 0):
                    MOx += PointsNode.getChild(i).trans.l2cam[0][3]/pointchild
                    MOy += PointsNode.getChild(i).trans.l2cam[1][3]/pointchild
                i+=1
            if(ChildMeadian == 0):
                pointchild += 1
                DynamicVariable = "Point" + str(pointchild)
                vars()[DynamicVariable]: GameObjectEntity_Point = PointSpawn(DynamicVariable,1,0,0)
                scene.world.addEntityChild(PointsNode, vars()[DynamicVariable])
                ChildMeadian = pointchild
            PointsNode.getChild(ChildMeadian).trans.l2cam = util.translate(MOx, MOy, 0)
            scene.world.traverse_visit(initUpdate, scene.world.root)
            time.sleep(0.15)
            i=1
        #Connect line between two points mechanism
        elif (key.is_pressed("s")):
            
            if(linechildZ == 0 and pointchild != 0):
                for i in range(len(keys)):
                    ValuesarrayLength = len(values[i])
                    linechildZCurrPlane=0
                    while(linechildZCurrPlane < ValuesarrayLength-1):
                        r = random.uniform(0, 1.0)
                        g = random.uniform(0, 1.0)
                        b = random.uniform(0, 1.0)
                        linechild +=1
                        linechildZ +=1
                        linechildZCurrPlane += 1
                        DynamicVariable = "Line" + str(linechild)
                        point1 =  values[i][linechildZCurrPlane][xPlane], values[i][linechildZCurrPlane][yPlane] , values[i][linechildZCurrPlane][zPlane], 1 
                        point2 =  values[i][linechildZCurrPlane-1][xPlane], values[i][linechildZCurrPlane-1][yPlane] , values[i][linechildZCurrPlane-1][zPlane], 1 

                        vars()[DynamicVariable]: GameObjectEntity = LineSpawn(DynamicVariable,point2,point1,r,g,b)
                        scene.world.addEntityChild(LinesNode, vars()[DynamicVariable])      
                scene.world.traverse_visit(initUpdate, scene.world.root)    
            else:
                toggleLineSwitch = not toggleLineSwitch
            
            time.sleep(0.15)
        #Line
        elif (key.is_pressed("l")):
            if(linechildPlane==0 and linechildZ != 0):
                for i in range(len(keys)):
                    LinesList += values[i]
                LinesList.sort(key = lambda row: (row[xPlane]),reverse=False)

                while(linechildPlane<pointchild -1):
                    linechild +=1 
                    linechildPlane +=1
                    DynamicVariable = "Line" + str(linechild)
                    point1 =  LinesList[linechildPlane][xPlane], LinesList[linechildPlane][yPlane] , LinesList[linechildPlane][zPlane], 1 
                    point2 =  LinesList[linechildPlane-1][xPlane], LinesList[linechildPlane-1][yPlane] , LinesList[linechildPlane-1][zPlane], 1 
                    vars()[DynamicVariable]: GameObjectEntity = LineSpawn(DynamicVariable,point2,point1,1,0,0)
                    scene.world.addEntityChild(LinesNode, vars()[DynamicVariable])
                scene.world.traverse_visit(initUpdate, scene.world.root)
            else:
                toggleLineZSwitch = not toggleLineZSwitch         
                
            time.sleep(0.15)
        #Super Function
        elif (key.is_pressed("f")):
            
            time.sleep(0.15)
        #QUIT button
        elif (key.is_pressed("q")):
            print(" x:", xPlane, " y: ", yPlane, " z: ", zPlane)
            break
        #enlarge points
        elif (key.is_pressed("x")):
            PointSize +=1
            gl.glPointSize(PointSize)
            time.sleep(0.15)
        #make points smaller
        elif (key.is_pressed("z")):
            PointSize -=1
            gl.glPointSize(PointSize)
            time.sleep(0.15)
        Display()
        
    scene.shutdown()
    

if __name__ == "__main__":    
    main(imguiFlag = True)


