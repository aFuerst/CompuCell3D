/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

 #include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/ClassRegistry.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
// // // #include <CompuCell3D/Field3D/Field3DIO.h>
#include <CompuCell3D/plugins/SimpleClock/SimpleClockPlugin.h>



using namespace CompuCell3D;


// // // #include <string>
// // // #include <fstream>
// // // #include <iostream>
// // // #include <sstream>
using namespace std;



#include "ChemotaxisDictyPlugin.h"


ChemotaxisDictyPlugin::ChemotaxisDictyPlugin() : field(0), potts(0), lambda(lambda),gotChemicalField(false),xmlData(0) {
}

ChemotaxisDictyPlugin::~ChemotaxisDictyPlugin() {

}


void ChemotaxisDictyPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {




  xmlData=_xmlData;
  sim = simulator;
  potts = simulator->getPotts();
  
  
//   chemicalEnergy->setSimulatorPtr(simulator);
  potts->registerEnergyFunctionWithName(this,"ChemotaxisDicty");

  bool pluginAlreadyRegisteredFlagNeighbor;
  Plugin *plugin=Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlagNeighbor);
  if(!pluginAlreadyRegisteredFlagNeighbor)
      plugin->init(sim);


   bool pluginAlreadyRegisteredFlag;
   SimpleClockPlugin *simpleClockPlugin=(SimpleClockPlugin *)Simulator::pluginManager.get("SimpleClock",&pluginAlreadyRegisteredFlag); //this will load SurfaceTracker plugin if it is not already loaded
  if(!pluginAlreadyRegisteredFlag)
      simpleClockPlugin->init(sim);


  simpleClockAccessorPtr=simpleClockPlugin->getSimpleClockAccessorPtr();
  simulator->registerSteerableObject(this);

}
///will initialize chemotactic field here - need to deffer this after all steppables which contain field had been pre-initialzied
void ChemotaxisDictyPlugin::extraInit(Simulator *simulator) {
	
	update(xmlData,true);

   
	

}

void ChemotaxisDictyPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell){
   if(!newCell){
      concentrationField->set(pt,0.0);/// in medium we assume concentration 0
   }
}





void ChemotaxisDictyPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	//if(potts->getDisplayUnitsFlag()){
	//	Unit energyUnit=potts->getEnergyUnit();




	//	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
	//	if (!unitsElem){ //add Units element
	//		unitsElem=_xmlData->attachElement("Units");
	//	}

	//	if(unitsElem->getFirstElement("LambdaUnit")){
	//		unitsElem->getFirstElement("LambdaUnit")->updateElementValue(energyUnit.toString());
	//	}else{
	//		CC3DXMLElement * energyElem = unitsElem->attachElement("LambdaUnit",energyUnit.toString());
	//	}
	//}

	nonChemotacticTypeVector.clear();

	lambda=_xmlData->getFirstElement("Lambda")->getDouble();

	if(_xmlData->findElement("NonChemotacticType")){

		nonChemotacticTypeVector.push_back(_xmlData->getFirstElement("NonChemotacticType")->getByte());
	}
   chemicalFieldName=_xmlData->getFirstElement("ChemicalField")->getText();
	chemicalFieldSource=_xmlData->getFirstElement("ChemicalField")->getAttribute("Source");
	initializeField();
}

double ChemotaxisDictyPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {


   if(!gotChemicalField)
      return 0.0;
   
///cells move up the concentration gradient
float concentration=field->get(pt);
float neighborConcentration=field->get(potts->getFlipNeighbor());
float energy = 0.0;
unsigned char type;
bool chemotaxisDone=false;

   /// new cell has to be different than medium i.e. only non-medium cells can chemotact
   ///e.g. in chemotaxis only non-medium cell can move a pixel that either belonged to other cell or to medium
   ///but situation where medium moves to a new pixel is not considered a chemotaxis
   if(newCell && simpleClockAccessorPtr->get(newCell->extraAttribPtr)->flag){
      energy+=(neighborConcentration - concentration)*lambda;
      chemotaxisDone=true;
      
   }
 
  if(!chemotaxisDone && oldCell && simpleClockAccessorPtr->get(oldCell->extraAttribPtr)->flag){
      energy+=(neighborConcentration - concentration)*lambda;
      chemotaxisDone=true;
  }

   return energy;
}

double ChemotaxisDictyPlugin::getConcentration(const Point3D &pt) {
  if (!field) throw CC3DException("No chemical field has been initialized!");
  return field->get(pt);
}

void ChemotaxisDictyPlugin::initializeField(){

   if(!gotChemicalField){///this is only temporary solution will have to come up with something better
      ClassRegistry *classRegistry=sim->getClassRegistry();
      Steppable * steppable=classRegistry->getStepper(chemicalFieldSource);
      //if(chemicalFieldSource=="DiffusionSolverBiofilmFE"){			
         field=((DiffusableVector<float> *) steppable)->getConcentrationField(chemicalFieldName);
         gotChemicalField=true;
      //}
  
  if (!field) throw CC3DException("No chemical field has been initialized!");
  
  }
	//cerr<<"field="<<field<<" conc="<<field->get(Point3D(10,10,10))<<endl;
	
}




std::string ChemotaxisDictyPlugin::toString(){
  return  "ChemotaxisDicty";
}



std::string ChemotaxisDictyPlugin::steerableName(){
 return  toString();
}

