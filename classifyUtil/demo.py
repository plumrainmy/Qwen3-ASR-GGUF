import json
from datetime import datetime, timedelta, timezone

input = "[{\"content\":\"【图片】\",\"role\":\"customer\",\"time\":1775526818000},{\"content\":\"咋不行呢\",\"role\":\"customer\",\"time\":1775526821000},{\"content\":\"已经把钱打过去了\",\"role\":\"customer\",\"time\":1775526828000},{\"content\":\"他这个银行我用行号搜不到\",\"role\":\"customer\",\"time\":1775526841000},{\"content\":\"【图片】\",\"role\":\"customer\",\"time\":1775527381000},{\"content\":\"你点击识别有误，强制提交\",\"role\":\"agent\",\"time\":1775528281000},{\"content\":\"点击了\",\"role\":\"customer\",\"time\":1775528300000},{\"content\":\"【图片】\",\"role\":\"customer\",\"time\":1775528325000},{\"content\":\"过了10分钟了已经\",\"role\":\"customer\",\"time\":1775528332000},{\"content\":\"刷新还是这样\",\"role\":\"customer\",\"time\":1775528337000},{\"content\":\"提交了就行，一般对公支付比较慢，2~3个小时才能审核\",\"role\":\"agent\",\"time\":1775528399000},{\"content\":\"好的\",\"role\":\"customer\",\"time\":1775528550000}]"

UTC_PLUS_8 = timezone(timedelta(hours=8))


def main(input: str) -> dict:
    try:
        # 原始JSON字符串
        data_str = input
        # 将字符串转为列表对象
        data_list = json.loads(data_str)
        lines = []
        for item in data_list:
            if 'msg_content' in item:
                msg_content = json.loads(item['msg_content'])
                content = msg_content.get('content', '')
            else:
                content = item.get('content', '')
            if 'role' in item:
                role = '客户' if item.get('role') == 'customer' else '客服'
            elif 'from_user' in item:
                role = '客户' if item.get('from_user', '').startswith('wmOBB_') else '客服'
            else:
                role = '未知角色'
            time_str = ""
            if 'time' in item:
                time_str = datetime.fromtimestamp(item.get('time') / 1000, tz=timezone.utc).astimezone(
                    UTC_PLUS_8
                ).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{time_str} {role}: {content}".strip())
        return {
            'text': '\n'.join(lines)
        }
    except:
        return {
            'text': input
        }


if __name__ == '__main__':
    print(main(input))
