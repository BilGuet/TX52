#include <iostream>

#include"sph_system.h"
using namespace std;

/* host == CPU */
/* device == GPU */

int main()
{
sphSystem SPH; /* SPH system object */
bool bStart = true;
SPH.Setup ( bStart );
//SPH.InsertParticles();
//
cout<<"It's ok"<<endl;


SPH.displayParticles(0); /* save initial data and send to device */


SPH.RunSimulation (1); /* SPH computation */
SPH.TransferFromCUDA (); /*  transfert from device to host */
SPH.displayParticles(1); /* save data */
return 0;

}
