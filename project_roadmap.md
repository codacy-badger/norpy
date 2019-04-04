**Basic Project Roadmap:**
*Improve Testing Infrastructure and create a unified and sound interface*
All tests are on the testing infrastructure at the moment.
All components of norpy_hatchery are tested from different perspectives in there.
One thing that remains to be done is to clean the file up and avoid any duplication
of code to increase readability.


*Removing Occupation B from the backward procedure*
Occ_B ha been removed from the state space creation and from the function that
calculates immediate rewards. The same procedure has to be performed on the
backward induction and the function that calculates future rewards.




*Remaining clean up remove occ*
Fix potential style guide violations.


*NOT: Add human capital as a further state variable*
We first fully implement norpy with the setup at hand and then incorporate the
additional state variable.



*We now want to work ahead and use norpy as soon as possible in as a replacement for the crude RESPY adjustments in the SMM and MLE estimation and then add the additional state variable there.*
We use SME and MLE wrappers to estimate the model.
It is essential to mention that this infrastructure fully separates between simulation and
estimation which allows for more flexibility.