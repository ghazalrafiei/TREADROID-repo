# Prerequisites

* Python 3.6 with the following packages installed: `Appium-Python-Client, beautifulsoup4, Flask, Flask-RESTful, gensim, html5lib, lxml, numpy, requests, scipy, selenium`
* [Appium-Desktop 1.10.0](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fappium%2Fappium-desktop%2Freleases%2Ftag%2Fv1.10.0&sa=D&sntz=1&usg=AFQjCNGsfo5xiY2Qn-P-ML7NlhKu73FT1A)
* [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
* Download the [subject apks](https://drive.google.com/open?id=1wb9ODzqMfsRCLqU80QF-g_1IrF0r7-vj) and `git clone` this project.

# Getting Started
1. Install subject apps on an emulator
2. Start the emulator; Start Appium-desktop
3. Run `w2v_service.py` first to activate the background web service for similarity query (modify the path in source code pointing to `GoogleNews-vectors-negative300.bin`)
4. Run Explorerm.py with arguments: 
```
python3 Explorerm.py ${TRANSFER_ID}
```
TRANSFER_ID is the transfer id, APPIUM_PORT is the port used by Appium-Desktop (4723 by default), EMULATOR is the name of the emulator, e.g., 
```
python3 Explorer.py a21-a22-b21
```
It will start transferring the test case of `a21-Minimal` to `a22-Clear` List for the b21-Add task function. 

5. The source test cases are under test-repo/[CATEGORY]/[FUNCTIONALITY]/base/[APP_ID].json, e.g., `test-repo/a2/b21/base/a21.json`. The generated test cases for the target app is under generated/[APP_FROM]-/[APP_TO]-[FUNCTIONALITY].json, e.g., `test-repo/a2/b21/generated/a21-a22-b21.json`

6. `Evaluator.py` is used to evaluate the test cases generated in terms of precision, recall, etc.
Let's say we would like to transfer the test `a2/b21/base/a21.json` to the app `a22`. The GUI events in `a2/b21/base/a21.json` (source events) will be loaded with `base_from` option. The correct GUI events to test the same feature on `a22`, i.e., `a2/b21/base/a22.json` (target events), will be loaded with `base_to` option. 