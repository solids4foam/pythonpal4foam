/* License
    This program is part of pythonPal4Foam.

    This program is free software: you can redistribute it and/or modify 
    it under the terms of the GNU General Public License as published 
    by the Free Software Foundation, either version 3 of the License, 
    or (at your option) any later version.

    This program is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

    See the GNU General Public License for more details. You should have 
    received a copy of the GNU General Public License along with this 
    program. If not, see <https://www.gnu.org/licenses/>. 

   Application
    pythonPalIcoFoam

   Original solver
    icoFoam

   Modified by
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved

   Description
    Transient solver for incompressible, laminar flow of Newtonian fluids.
    The final results are sent to the Python interpreter for the calculation
    of the specific kinetic energy.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
#include "pythonPal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase2.H"
    #include "createTime.H"
    #include "createMesh.H"

    pisoControl piso(mesh);

    #include "createFields.foamextend.H"
    #include "initContinuityErrs.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    pythonPal myPythonPal("python_script.py", true);

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {

        Info<< "Time = " << runTime.timeName() << nl << endl;

        #include "CourantNo.H"

        // Time-derivative matrix
        fvVectorMatrix ddtUEqn(fvm::ddt(U));

        // Convection-diffusion matrix
        fvVectorMatrix HUEqn
        (
            fvm::div(phi, U)
          - fvm::laplacian(nu, U)
        );

        if (piso.momentumPredictor())
        {
            solve(ddtUEqn + HUEqn == -fvc::grad(p));
        }

        // Prepare clean 1/a_p without time derivative contribution
        volScalarField rAU = 1.0/HUEqn.A();

        // --- PISO loop
        while (piso.correct())
        {
            // Calculate U from convection-diffusion matrix
            U = rAU*HUEqn.H();

            // Consistently calculate flux
            piso.calcTransientConsistentFlux(phi, U, rAU, ddtUEqn);

            adjustPhi(phi, U, p);

            while (piso.correctNonOrthogonal())
            {
                fvScalarMatrix pEqn
                (
                    fvm::laplacian
                    (
                        fvc::interpolate(rAU)/piso.aCoeff(U.name()),
                        p,
                        "laplacian(rAU," + p.name() + ')'
                    )
                 ==
                    fvc::div(phi)
                );

                pEqn.setReference(pRefCell, pRefValue);
                pEqn.solve();

                if (piso.finalNonOrthogonalIter())
                {
                    phi -= pEqn.flux();
                }
            }

#           include "continuityErrs.H"

            // Consistently reconstruct velocity after pressure equation
            piso.reconstructTransientVelocity(U, phi, ddtUEqn, rAU, p);
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    //Pass the internal fields to Python side
    myPythonPal.passToPython(U, "U");
    myPythonPal.passToPython(k, "k");

    // Calculate k for the internal field
    myPythonPal.execute("k[:, :] = calculatek(U)");

    // Calculate k for all boundary patches
    forAll(U.boundaryField(), patchI)
    {
        if (U.boundaryField()[patchI].size() != 0)
        {
            // Pass the boundary fields to Python side
            myPythonPal.passToPython(U.boundaryField()[patchI], "U");
            myPythonPal.passToPython(k.boundaryField()[patchI], "k");

            // Calculate k for boundary patchI
            myPythonPal.execute("k[:, :] = calculatek(U)");
        }
    }

    k.write();

    //Next, we test the additional methods in pythonPal4foam

    // ***************************************************************
    // 1. Test for the passWordToPython method
    word message = "Thank you for using pythonPal4foam";

    // Pass the name of a message, given by a Foam::word, to Python
    myPythonPal.passWordToPython(message, "messageInPython"); 
    // Now, a variable "messageInPython" exists in Python 
    // ***************************************************************

    // ***************************************************************
    // 2. Test for the retrieveWordFromPython method. 
    //Declare a Foam:: word
    word result;

    // Retrieve whatever is saved in the "messageInPython" variable in Python
    result = myPythonPal.retrieveWordFromPython("messageInPython");

    //Print it to check it is what is expected
    Info << endl << "result is: " << result << endl;

    // 2.1. Modify the variable "messageInPython" via PythonPal
    myPythonPal.execute("messageInPython += '. We hope it has been useful.' ");

    // Print countryInPython
    myPythonPal.execute("print(messageInPython)");
    // ***************************************************************

    // ***************************************************************
    // 3. Test for the passScalarToPython method
    // Pass both a scalar and the name it will have in Python, to Python
    myPythonPal.passScalarToPython(2.0, "numberDevelopers"); 
    // Now, a variable "numberDevelopers" exists in Python 
    // ***************************************************************

    // ***************************************************************
    // 4. Test for the retrieveScalarToPython method
    // Retrieve whatever is saved in the "numberTeams" variable in Python
    scalar totalDevelopers = myPythonPal.retrieveScalarFromPython("numberDevelopers");

    InfoIn("retrieveScalarFromPython(...)")
    << "Total number of developers in pythonPal4Foam team is " << totalDevelopers << endl;
    // ***************************************************************

    return 0;
}


// ************************************************************************* //
