import json
import time
import logging
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from aliyunsdkcore.auth.credentials import AccessKeyCredential
from app.config import config

logger = logging.getLogger("AliyunFileTrans")

class AliyunFileTrans:
    REGION_ID   = config.get("ALIYUN_REGION_ID", "cn-shanghai")
    DOMAIN      = config.get("ALIYUN_DOMAIN", "filetrans.cn-shanghai.aliyuncs.com")
    API_VERSION = config.get("ALIYUN_API_VERSION", "2018-08-17")
    POST_ACTION = "SubmitTask"
    GET_ACTION  = "GetTaskResult"

    STATUS_SUCCESS = "SUCCESS"
    STATUS_RUNNING = "RUNNING"
    STATUS_QUEUEING = "QUEUEING"
    STATUS_SUCCESS_WITH_NO_VALID_FRAGMENT = "SUCCESS_WITH_NO_VALID_FRAGMENT"

    def __init__(self, access_key_id, access_key_secret):
        self.logger = logging.getLogger("AliyunFileTrans")
        credentials = AccessKeyCredential(access_key_id, access_key_secret)
        self.client = AcsClient(region_id=self.REGION_ID, credential=credentials)

    def submit_file_trans_request(self, app_key, file_link):
        req = CommonRequest()
        req.set_domain(self.DOMAIN)
        req.set_version(self.API_VERSION)
        req.set_action_name(self.POST_ACTION)
        req.set_method("POST")

        task_object = {
            "appkey": app_key,
            "file_link": file_link,
            "version": "4.0",
            "enable_words": True,
            "sample_rate": 16000,
            "channel_num": 1
        }
        req.add_body_params('Task', json.dumps(task_object))

        try:
            resp = self.client.do_action_with_exception(req)
            result = json.loads(resp.decode('utf-8'))
            status_text = result.get("StatusText")
            if status_text == self.STATUS_SUCCESS:
                return result.get("TaskId")
            logger.error(f"提交请求失败: {status_text}")
        except Exception as e:
            logger.error(f"提交请求异常: {e}")
        return None

    def get_file_trans_result(self, task_id):
        req = CommonRequest()
        req.set_domain(self.DOMAIN)
        req.set_version(self.API_VERSION)
        req.set_action_name(self.GET_ACTION)
        req.set_method("GET")
        req.add_query_param('TaskId', task_id)

        result = None
        retry = 0
        while True:
            try:
                retry += 1
                resp = self.client.do_action_with_exception(req)
                root = json.loads(resp.decode('utf-8'))
                status_text = root.get("StatusText")
                logger.info(f"查询状态: {status_text} (#{retry})")
                if status_text in (self.STATUS_RUNNING, self.STATUS_QUEUEING):
                    time.sleep(10)
                    continue
                if status_text in (self.STATUS_SUCCESS, self.STATUS_SUCCESS_WITH_NO_VALID_FRAGMENT):
                    result = root.get("Result") or ""
                else:
                    logger.error(f"识别失败: {status_text}")
                break
            except Exception as e:
                logger.error(f"查询异常: {e}")
                break
        return result

    def format_result_for_mq(self, result_json):
        if not result_json:
            return {"text":"", "segments":[], "language":"zh"}
        try:
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
            sentences = result.get("Sentences", []) or []
            text = " ".join(s.get("Text","") for s in sentences)
            segs = []
            for i,s in enumerate(sentences):
                seg = {
                    "id": i,
                    "start": int(s.get("BeginTime",0)),
                    "end": int(s.get("EndTime",0)),
                    "text": s.get("Text","")
                }
                if "Words" in s:
                    words = []
                    for w in s["Words"]:
                        words.append({
                            "start": int(w.get("BeginTime",0)),
                            "end": int(w.get("EndTime",0)),
                            "word": w.get("Word","")
                        })
                    seg["words"] = words
                segs.append(seg)
            return {"text": text, "segments": segs, "language":"zh"}
        except Exception as e:
            logger.error(f"格式化异常: {e}")
            return {"text":"", "segments":[], "language":"zh"}
