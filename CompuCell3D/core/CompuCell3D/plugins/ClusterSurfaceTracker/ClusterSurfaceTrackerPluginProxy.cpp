#include "ClusterSurfaceTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto clusterSurfaceTrackerProxy = registerPlugin<Plugin, ClusterSurfaceTrackerPlugin>(
        "ClusterSurfaceTracker",
        "Autogenerated plugin - the author of the plugin should provide brief description here",
        &Simulator::pluginManager
);
