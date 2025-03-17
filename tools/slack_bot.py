#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
slack_bot.py
Created on May 02 2020 11:02
a bot to send message/image during program run
@author: Tu Bui tu@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import requests
import socket
from slack import WebClient
from slack.errors import SlackApiError
import threading


SLACK_MAX_PRINT_ERROR = 3
SLACK_ERROR_CODE = {'not_active': 1,
                    'API': 2}


def welcome_message():
    hostname = socket.gethostname()
    all_args = ' '.join(sys.argv)
    out_text = 'On server {}: {}\n'.format(hostname, all_args)
    return out_text


class Notifier(object):
    """
    A slack bot to send text/image to a given workspace channel.
    This class initializes with a text file as input, the text file should contain 2 lines:
        slack token
        slack channel

    Usage:
    msg = Notifier(token_file)
    msg.send_initial_text(' '.join(sys.argv))
    msg.send_text('hi, this text is inside slack thread')
    msg.send_file(your_file, 'file title')
    """
    def __init__(self, token_file):
        """
        setup slack
        :param token_file: path to slack token file
        """
        self.active = True
        self.thread_id = None
        self.counter = 0  # count number of errors during Web API call
        if not os.path.exists(token_file):
            print('[SLACK] token file not found. You will not be notified.')
            self.active = False
        else:
            try:
                with open(token_file, 'r') as f:
                    lines = f.readlines()
                self.token = lines[0].strip()
                self.channel = lines[1].strip()
            except Exception as e:
                print(e)
                print('[SLACK] fail to read token file. You will not be notified.')
                self.active = False

    def _handel_error(self, e):
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        self.counter += 1
        if self.counter <= SLACK_MAX_PRINT_ERROR:
            print(f"Got the following error, you will not be notified: {e.response['error']}")

    def send_init_text(self, text=None):
        """
        start a new thread with a main message and register the thread id
        :param text: initial message for this thread
        :return:
        """
        if not self.active:
            return SLACK_ERROR_CODE['not_active']
        try:
            if text is None:
                text = welcome_message()
            sc = WebClient(self.token)
            response = sc.chat_postMessage(channel=self.channel, text=text)
            self.thread_id = response['ts']
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']
        print('[SLACK] sent initial text. Chat ID %s. Message %s' % (self.thread_id, text))
        return 0

    def send_init_file(self, file_path, title=''):
        """
        start a new thread with a file and register thread id
        :param file_path: path to file
        :param title: title of this file
        :return: 0 if success otherwise error code
        """
        if not self.active:
            return SLACK_ERROR_CODE['not_active']
        try:
            response = sc.files_upload(title=title, channels=self.channel, file=file_path)
            self.thread_id = response['ts']
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']
        print('[SLACK] sent initial file. Chat ID %s.' % self.thread_id)
        return 0

    def send_text(self, text, reply_broadcast=False):
        """
        send text as a thread if one is registered in self.thread_id.
        Otherwise send as a new message
        :param text: message to send.
        :return: 0 if success, error code otherwise
        """
        print(text)
        if not self.active:
            return SLACK_ERROR_CODE['not_active']
        if self.thread_id is None:
            self.send_init_text(text)
        else:
            try:
                sc = WebClient(self.token)
                response = sc.chat_postMessage(channel=self.channel, text=text,
                                               thread_ts=self.thread_id, as_user=True,
                                               reply_broadcast=reply_broadcast)
            except SlackApiError as e:
                self._handel_error(e)
                return SLACK_ERROR_CODE['API']
        return 0

    def _send_file(self, file_path, title='', reply_broadcast=False):
        """can be multithread target"""
        try:
            sc = WebClient(self.token)
            sc.files_upload(title=title, channels=self.channel,
                            thread_ts=self.thread_id, file=file_path,
                            reply_broadcast=reply_broadcast)
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']
        return 0

    def send_file(self, file_path, title='', reply_broadcast=False):
        if not self.active:
            return SLACK_ERROR_CODE['not_active']
        if self.thread_id is None:
            return self.send_init_file(file_path, title)
        else:
            os_thread = threading.Thread(target=self._send_file, args=(file_path, title, reply_broadcast))
            os_thread.start()
        return 0  # may still have error if _send_file() fail