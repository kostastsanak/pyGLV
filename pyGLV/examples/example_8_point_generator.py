

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
    |                                                      |-TriangleNode
    |                                                      |-------------|-----------|-------------|          
    |---------------------------|                          trans7        mesh7      shaderdec7      vArray7
    entityCam1,                 PointsNode,      
    |-------|                    |--------------|----------|--------------|           
    trans1, entityCam2           trans4,        mesh4,     shaderDec4     vArray4
            |                               
            ortho, trans2                   

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

def PointSpawn(pointname = "Point",r=0.7,g=0.7,b=0.7): 
    point = GameObjectEntity(pointname,primitiveID=gl.GL_POINTS)
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
    scene = Scene()    

    # Scenegraph with Entities, Components
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

    TriangleNode = scene.world.createEntity(Entity("TriangleNode"))
    scene.world.addEntityChild(rootEntity, TriangleNode)
    trans7 = BasicTransform(name="trans7", trs=util.identity())
    scene.world.addComponent(TriangleNode, trans7)

    axes = scene.world.createEntity(Entity(name="axes"))
    scene.world.addEntityChild(rootEntity, axes)
    axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
    axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))

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
    index = np.array((0,1,2), np.uint32) #simple triangle
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
    scene.init(imgui=True, windowWidth = 1024, windowHeight = 768, windowTitle = "pyglGA test_renderAxesTerrainEVENT", customImGUIdecorator = ImGUIecssDecorator)
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


    updateTRS = Event(name="OnUpdateTRS", id=100, value=None) #lines 255-258 contains the Scenegraph in a GUI as is it has issues.. To be fixed
    updateBackground = Event(name="OnUpdateBackground", id=200, value=None)
    eManager._events[updateTRS.name] = updateTRS
    eManager._events[updateBackground.name] = updateBackground


    eManager._subscribers[updateTRS.name] = gGUI
    eManager._subscribers[updateBackground.name] = gGUI

    eManager._subscribers['OnUpdateWireframe'] = gWindow
    eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
    eManager._subscribers['OnUpdateCamera'] = gWindow 
    eManager._actuators['OnUpdateCamera'] = renderGLEventActuator
    # MANOS END
    # Add RenderWindow to the EventManager publishers
    # eManager._publishers[updateBackground.name] = gGUI
    eManager._publishers[updateBackground.name] = gGUI ## added


    eye = util.vec(5.0, 4.0, 3.5)
    target = util.vec(0.0, 0.0, 0.0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.lookat(eye, target, up)
    
    projMat = util.perspective(50.0, 1.0, 1.0, 20.0) ## Setting the camera as perspective 

    gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

    model_terrain_axes = util.translate(0.0,0.0,0.0)
    
    #global point size integer
    PointSize = 5    
    #LISTS
    LinesList = [] 
    SplineList = []
    keys = []
    values = []
    #LISTS
    toggleSwitch = True
    toggleLineSwitch = True
    toggleLineZSwitch = True
    toggleSplineSwitch = True
    togglePlatformSwitch = True
    #MOST USEFUL GLOBALS
    pointchild = 0
    linechild = 0
    splinechild = 0
    trianglechild =0
    #ALWAYS CHANGING GLOBALS
    linechildZ = 0#used for z=0 lines
    linechildPlane = 0#used for lines that interconnect z=0 with z=1
    ChildMeadian = 0#used for median point
    i=1#used for object(points/lines) printing inside the while loop


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
    
    #we split our csv based on common Z plane values and pass it to 2 lists for later use
    for Zplanekey, value in itertools.groupby(pointListfromCSV, lambda x: x[zPlane]):
        keys.append(Zplanekey)
        values.append(list(value))
    for i in range(len(keys)):
        values[i].sort(key = lambda row: (row[xPlane]),reverse=False)
    
    gl.glPointSize(PointSize)
    #Displays all nodes created
    def Dispaly():
        i=1
        #print points
        while i<=pointchild:
            PointsNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @ PointsNode.getChild(i).trans.l2cam, mat4=True)
            PointsNode.getChild(i).shaderDec.setUniformVariable(key='my_color', value=[1, 0, 0], float4=True) #its porpuse is to change color on my_color vertex by setting the uniform                  

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
        #print Splines
        while i <= splinechild:
            if(toggleSplineSwitch):
                SplineNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @SplineNode.getChild(i).trans.l2cam, mat4=True)
            else:
                SplineNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1
            
        i=1
        #print platform
        while i <= trianglechild:
            if togglePlatformSwitch:
                TriangleNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=mvp_point @TriangleNode.getChild(i).trans.l2cam, mat4=True)
            else:
                TriangleNode.getChild(i).shaderDec.setUniformVariable(key='modelViewProj', value=None, mat4=True)
            i+=1

        i=1
        scene.render_post()
    while running:
        running = scene.render(running)
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        view =  gWindow._myCamera # updates view via the imgui
        mvp_point = projMat @ view 
        mvp_terrain_axes = projMat @ view @ model_terrain_axes
        
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
            if (pointchild == 0 ):
                for i in range(len(pointListfromCSV)):
                    pointchild += 1
                    DynamicVariable = "Point" + str(pointchild)
                    vars()[DynamicVariable]: GameObjectEntity = PointSpawn(DynamicVariable,0,1,1)
                    scene.world.addEntityChild(PointsNode, vars()[DynamicVariable]) 
                    PointsNode.getChild(pointchild).trans.l2cam =  util.translate(pointListfromCSV[i][xPlane], pointListfromCSV[i][yPlane], pointListfromCSV[i][zPlane])
                scene.world.traverse_visit(initUpdate, scene.world.root)
            
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
                vars()[DynamicVariable]: GameObjectEntity = PointSpawn(DynamicVariable,1,0,0)
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
        #print spline
        elif (key.is_pressed("e")):
            if(splinechild ==0):
                for i in range(len(keys)):
                    if(values[i]):
                        arr = np.array(values[i])
                        xPointToSpline = arr[:,xPlane]
                        yPointToSpline = arr[:,yPlane]
                        
                        f = CubicSpline(xPointToSpline,yPointToSpline, bc_type='natural')
                        x_new = np.linspace(0.2, 3, len(values[i])*6)
                        y_new = f(x_new)
                        z_new = keys[i]
                        l =0
                        while l < len(x_new) -1 :
                            splinechild += 1
                            l += 1
                            DynamicVariable = "Spline" + str(splinechild)
                            point1 = x_new[l], y_new[l], z_new, 1
                            point2 = x_new[l-1], y_new[l-1], z_new, 1  
                            vars()[DynamicVariable]: GameObjectEntity = LineSpawn(DynamicVariable,point2,point1,0,1,0)
                            scene.world.addEntityChild(SplineNode, vars()[DynamicVariable])
                        scene.world.traverse_visit(initUpdate, scene.world.root)
                        SplineList.append(f)#pass the spline functions to a list for later use
            else:
                toggleSplineSwitch = not toggleSplineSwitch
            
            time.sleep(0.15)
        elif(key.is_pressed("b")):
            if(SplineList and trianglechild==0):
                spline1 = SplineList[0]
                spline2 = SplineList[1]
                x_new = np.linspace(0.2, 3, len(values[i])*6)

                y_spline1 = spline1(x_new)
                y_spline2 = spline2(x_new)
                z_spline1 = keys[0]
                z_spline2 = keys[1]
                l =0
                while l < len(x_new) -1 :
                    l += 1
                    #first triangle
                    trianglechild += 1
                    DynamicVariable = "Triangle" + str(trianglechild)
                    point1 = x_new[l-1], y_spline1[l-1], z_spline1, 1
                    point2 = x_new[l], y_spline1[l], z_spline1, 1
                    point3 = x_new[l-1], y_spline2[l-1], z_spline2, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3)
                    scene.world.addEntityChild(TriangleNode, vars()[DynamicVariable])
                    #second triangle
                    trianglechild += 1
                    DynamicVariable = "Triangle" + str(trianglechild)
                    point1 = x_new[l-1], y_spline2[l-1], z_spline2, 1
                    point2 = x_new[l], y_spline2[l], z_spline2, 1
                    point3 = x_new[l], y_spline1[l], z_spline1, 1
                    vars()[DynamicVariable]: GameObjectEntity = TriangleSpawn(DynamicVariable,point1,point2,point3)
                    scene.world.addEntityChild(TriangleNode, vars()[DynamicVariable])
                scene.world.traverse_visit(initUpdate, scene.world.root)
            else:
                togglePlatformSwitch = not togglePlatformSwitch
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
        Dispaly()
        
    scene.shutdown()
    

if __name__ == "__main__":    
    main(imguiFlag = True)


