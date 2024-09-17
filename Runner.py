# from distutils.command.config import config
from appium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from appium.webdriver.common.appiumby import AppiumBy as MobileBy
from appium.webdriver.common.touch_action import TouchAction
from appium.options.android import UiAutomator2Options
import subprocess
from selenium.common.exceptions import NoSuchElementException
import time
import re
from subprocess import call
# local import
from Databank import Databank
# from misc import teardown_mail
from StrUtil import StrUtil


class Runner:
    def __init__(self, pkg, act, no_reset=False, appium_port='4723', udid=None):
        desired_caps = Runner.set_caps(pkg, act, no_reset, udid)
        capabilities_options = UiAutomator2Options().load_capabilities(desired_caps)
        self.driver = webdriver.Remote(command_executor='http://localhost:' + appium_port, options=capabilities_options)

        self.databank = Databank()
        self.act_interval = 2

    @staticmethod
    def set_caps(app_name, app_activity, no_reset=False, udid=None):
        caps = {
            'platformName': 'Android',
            'platformVersion': '12.0',
            'deviceName': 'Android Emulator',
            'automationName':"uiautomator2",
            'appPackage': app_name,
            'appActivity': app_activity,
            'autoGrantPermissions': True,
            'newCommandTimeout': 6000,
            'noReset': True,
            "appWaitForLaunch" : False
        }
        if udid:
            caps['udid'] = udid
        return caps

    def perform_actions(self, action_list, require_wait=False, reset=True, cgp=None):
        if reset:
            if self.driver.desired_capabilities['desired']['noReset']:
                # self.driver.launch_app() is deprecated
                self.driver.activate_app(app_id=self.driver.desired_capabilities['appPackage'])  # don't clear app data
            else:
                # self.driver.reset() is deprecated
                self.driver.terminate_app(app_id=self.driver.desired_capabilities['appPackage'])
                subprocess.run(f"adb shell pm clear {self.driver.desired_capabilities['appPackage']}".split(),
                                stdout=subprocess.DEVNULL)
                self.driver.activate_app(app_id=self.driver.desired_capabilities['appPackage'])

        is_for_confirm = False
        # specific for Yelp app. Cancel the pop-up dialog
        try:
            ele = self.driver.find_element_by_id('com.yelp.android:id/toolbar')
            ele.click()
        except:
            pass
        print("action_list",action_list)
        for i, action in enumerate(action_list):
            time.sleep(self.act_interval)
            # if the action is SYS_EVENT, no need to get the element
            if action['class'] == 'SYS_EVENT':
                if action['action'][0] == 'sleep':
                    time.sleep(action['action'][1])
                elif action['action'][0] == 'KEY_BACK':
                    self.driver.press_keycode(4)  # AndroidKeyCode for 'Back'
                elif action['action'][0] == 'restart_app':
                    self.driver.activate_app(self.driver.desired_capabilities['appPackage'])
                else:
                    assert False, 'Unknown SYS_EVENT'
                continue

            if action['class'] == 'EMPTY_EVENT':
                continue

            self.hide_keyboard()
            # if the action is WAIT_UNTIL, no need to get the element
            if action['action'][0].startswith('wait_until'):
                # e.g., ["wait_until_element_presence", 10, "xpath", "//android.widget.TextView[@text='Sample Todo']"]
                # e.g., ["wait_until_element_invisible", 10, "xpath", "//android.widget.TextView[@text='Sample Todo']"]
                # e.g., ["wait_until_text_presence", 10, "text", "65.09"]
                # e.g., ["wait_until_text_invisible", 10, "text", "Sample Todo"]
                wait_time, selector_type, selector = action['action'][1:]
                locator = None
                if selector_type == 'xpath':
                    locator = (MobileBy.XPATH, selector)
                elif selector_type == 'content-desc':
                    locator = (MobileBy.ACCESSIBILITY_ID, selector)
                elif selector_type == 'id':
                    locator = (MobileBy.ID, selector)
                elif selector_type == 'text':
                    locator = (MobileBy.XPATH, f'//*[contains(@text, "{selector}")]')
                else:
                    assert locator, "Unknown selector type"
                try:
                    if action['action'][0].endswith('presence'):
                        WebDriverWait(self.driver, wait_time).until(EC.presence_of_element_located(locator))
                    elif action['action'][0].endswith('invisible'):
                        WebDriverWait(self.driver, wait_time).until(EC.invisibility_of_element_located(locator))
                    else:
                        assert False, "Unknown WAIT_UNTIL action"
                except Exception as excep:
                    print('Exception in wait_until')
                    print(excep)
                    print(action)
                    print(locator)
                    print(self.driver.page_source)
                    assert False, "Failed WAIT_UNTIL action"
                continue

            # action performed on the selected element
            ele = self.get_web_element(action)
            act_from = self.get_current_package() + self.get_current_activity()
            if ele:
                if action['action'][0] == 'click':
                    # specific corner case for Yelp: click the right part
                    if 'activity_login_create_account_question' in action['resource-id']\
                            and action['text'] == "Don't have a Yelp account yet? Sign up.":
                        rect = ele.rect
                        x = rect['x'] + (0.8 * rect['width'])
                        y = rect['y'] + (0.5 * rect['height'])
                        self.driver.tap([(x, y)])
                    else:
                        ele.click()
                elif 'send_keys' in action['action'][0]:
                    value_for_input = action['action'][1]
                    # if sending email (for registration), get a new one
                    print("value_for_input",value_for_input)
                    if not action['resource-id'].endswith('to'):
                        # print("11")
                        if StrUtil.is_contain_email(value_for_input) and value_for_input not in self.databank.get_login_email():
                            if is_for_confirm:
                                value_for_input = self.databank.get_temp_email(renew=False)
                            else:
                                value_for_input = self.databank.get_temp_email()
                                is_for_confirm = True
                    # all possible cases: 'clear_and_send_keys', 'clear_and_send_keys_and_hide_keyboard',
                    # 'send_keys_and_hide_keyboard', 'send_keys_and_enter', 'send_keys'
                    if action['action'][0].startswith('clear'):
                        ele.clear()
                    ele.click()
                    ele.send_keys(value_for_input)
                    if action['action'][0].endswith('hide_keyboard'):
                        time.sleep(self.act_interval/2)
                        ele.click()
                        self.hide_keyboard()
                    elif action['action'][0].endswith('enter'):
                        self.driver.press_keycode(66)  # AndroidKeyCode for 'Enter'

                elif action['action'][0] == 'swipe_right':
                    rect = ele.rect  # e.g., {'x': 202, 'y': 265, 'width': 878, 'height': 57}
                    start_x, start_y, end_x, end_y = rect['x'] + rect['width'] / 4, rect['y'] + rect['height'] / 2, \
                                                     rect['x'] + rect['width'] * 3 / 4, rect['y'] + rect['height'] / 2
                    self.driver.swipe(start_x, start_y, end_x, end_y, 500)
                elif action['action'][0] == 'long_press':
                    ta = TouchAction(self.driver)
                    ta.long_press(ele).perform()
                else:
                    assert False, "Unknown action to be performed"
                act_to = self.get_current_package() + self.get_current_activity()
                if action['action'][0] in ['click', 'long_press'] and cgp:
                    cgp.add_edge(act_from, act_to, action)

        if require_wait:
            time.sleep(self.act_interval*2)
        else:
            # time.sleep(self.act_interval/2)
            time.sleep(self.act_interval)

    def get_web_element(self, action):
        ele = None
        try:
            xpath = None
            if action['resource-id']:
                if 'id-prefix' in action and '/' not in action['resource-id']:  # for running actions when exploring
                    rid = action['id-prefix'] + action['resource-id']
                else:
                    rid = action['resource-id']  # for running actions load from test file
                elements = self.driver.find_elements(MobileBy.ID, rid)
                if elements:
                    ele = elements[0]
                    if len(elements) > 1:
                        if action['text'] or action['content-desc']:
                            attr = 'text' if action['text'] else 'content-desc'
                            xpath = f'//{action["class"]}[contains(@{attr}, "{action[attr]}") ' \
                                f'and @resource-id="{rid}"]'
                            ele = self.driver.find_element(MobileBy.XPATH, xpath)
            elif action['content-desc']:
                xpath = '//' + action['class'] + '[@content-desc="' + action['content-desc'] + '"]'
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((MobileBy.XPATH, xpath)))
                ele = self.driver.find_element(MobileBy.XPATH, xpath)
            elif action['text']:
                xpath = '//' + action['class'] + '[@text="' + action['text'] + '"]'
                ele = self.driver.find_element(MobileBy.XPATH, xpath)
            elif action['naf']:  # "naf" is either "true" or ""; a32-a33-b31
                xpath = '//' + action['class'] + '[@NAF="true"]'
                ele = self.driver.find_element(MobileBy.XPATH, xpath)
            else:
                assert False, "No attribute to locate widgets"
        except Exception as excep:
            print('Exception in get_web_element')
            print(excep)
            print(action)
            if xpath:
                print(xpath)
            print(self.driver.page_source)
        return ele



    def check_text_invisible(self, act):
        # e.g., ["wait_until_text_invisible", 10, "text", "Sample Todo"]
        wait_time, selector_type, selector = act['action'][1:]
        assert selector_type == 'text'
        try:
            WebDriverWait(self.driver, wait_time).until(
                EC.invisibility_of_element_located((MobileBy.XPATH, f'//*[contains(@text, "{selector}")]')))
            return True
        except:
            return False

    def get_current_activity(self):
        return self.driver.current_activity

    def get_page_source(self):
        self.hide_keyboard()
        return self.driver.page_source

    def get_current_package(self):
        return self.driver.current_package

    def hide_keyboard(self):
        if self.driver.is_keyboard_shown:
            try:
                self.driver.hide_keyboard()
            except WebDriverException:
                pass


