""" Deep Learning Telegram bot
DLBot and TelegramBot Callback classes for the monitoring and control
of a Keras \ Tensorflow training process using a Telegram bot
By: Eyal Zakkay, 2019
https://eyalzk.github.io/
"""

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)

import numpy as np
from keras.callbacks import Callback
import keras.backend as K
import logging
from io import BytesIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class DLBot(object):
    """  A class for interacting with a Telegram bot to monitor and control a Keras \ tensorflow training process.
    Supports the following commands:
     /start: activate automatic updates every epoch and get a reply with all command options
     /help: get a reply with all command options
     /status: get a reply with the latest epoch's results
     /getlr: get a reply with the current learning rate
     /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
     /plot: get a reply with the loss convergence plot image
     /quiet: stop getting automatic updates each epoch
     /stoptraining: kill training process

    # Arguments
        token: String, a telegram bot token
        user_id: Integer. Specifying a telegram user id will filter all incoming
                 commands to allow access only to a specific user. Optional, though highly recommended.
    """


    def __init__(self, token, user_id=None):
        self.token = token  # bot token
        self.user_id = user_id  # id of the user with access
        self.filters = None
        self.chat_id = None  # chat id, will be fetched during /start command
        self.bot_active = False  # currently not in use
        self._status_message = "No status message was set"  # placeholder status message
        self.lr = None
        self.modify_lr = 1.0  # Initial lr multiplier
        self.verbose = True   # Automatic per epoch updates
        self.stop_train_flag = False  # Stop training flag
        self.updater = None
        # Initialize loss monitoring
        self.loss_hist = []
        self.val_loss_hist = []
        # Enable logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Message to display on /start and /help commands
        self.startup_message = "Hi, I'm the DL bot! I will send you updates on your training process.\n" \
                               " send /start to activate automatic updates every epoch\n" \
                               " send /help to see all options.\n" \
                               " Send /status to get the latest results.\n" \
                               " Send /getlr to query the current learning rate.\n" \
                               " Send /setlr to change the learning rate.\n" \
                               " Send /quiet to stop getting automatic updates each epoch\n" \
                               " Send /plot to get a loss convergence plot.\n" \
                               " Send /stoptraining to stop training process.\n\n"

    def activate_bot(self):
        """ Function to initiate the Telegram bot """
        self.updater = Updater(self.token)  # setup updater
        dp = self.updater.dispatcher  # Get the dispatcher to register handlers
        dp.add_error_handler(self.error)  # log all errors

        self.filters = Filters.user(user_id=self.user_id) if self.user_id else None
        # Command and conversation handles
        dp.add_handler(CommandHandler("start", self.start, filters=self.filters))  # /start
        dp.add_handler(CommandHandler("help", self.help, filters=self.filters))  # /help
        dp.add_handler(CommandHandler("status", self.status, filters=self.filters))  # /get status
        dp.add_handler(CommandHandler("getlr", self.get_lr, filters=self.filters))  # /get learning rate
        dp.add_handler(CommandHandler("quiet", self.quiet, filters=self.filters))  # /stop automatic updates
        dp.add_handler(CommandHandler("plot", self.plot_loss, filters=self.filters))  # /plot loss
        dp.add_handler(self.lr_handler())  # set learning rate
        dp.add_handler(self.stop_handler())  # stop training

        # Start the Bot
        self.updater.start_polling()
        self.bot_active = True

        # Uncomment next line while debugging
        # updater.idle()

    def stop_bot(self):
        """ Function to stop the bot """
        self.updater.stop()
        self.bot_active = False

    def start(self, bot, update):
        """ Telegram bot callback for the /start command.
        Fetches chat_id, activates automatic epoch updates and sends startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id
        self.verbose = True

    def help(self, bot, update):
        """ Telegram bot callback for the /help command. Replies the startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id

    def quiet(self, bot, update):
        """ Telegram bot callback for the /quiet command. Stops automatic epoch updates"""
        self.verbose = False
        update.message.reply_text(" Automatic epoch updates turned off. Send /start to turn epoch updates back on.")

    def error(self, update, error):
        """Log Errors caused by Updates."""
        self.logger.warning('Update "%s" caused error "%s"', update, error)

    def send_message(self,txt):
        """ Function to send a Telegram message to user
         # Arguments
            txt: String, the message to be sent
        """
        assert isinstance(txt, str)
        if self.chat_id is not None:
            self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
        else:
            print('Send message failed, user did not send /start')

    def set_status(self,txt):
        """ Function to set a status message to be returned by the /status command """
        self._status_message = txt

    def status(self, bot, update):
        """ Telegram bot callback for the /status command. Replies with the latest status"""
        update.message.reply_text(self._status_message)

    # Setting Learning Rate Callbacks:
    def get_lr(self, bot, update):
        """ Telegram bot callback for the /getlr command. Replies with current learning rate"""
        if self.lr:
            update.message.reply_text("Current learning rate: " + str(self.lr))
        else:
            update.message.reply_text("Learning rate was not passed to DL-Bot")

    def set_lr_front(self, bot, update):
        """ Telegram bot callback for the /setlr command. Displays option buttons for learning rate multipliers"""
        reply_keyboard = [['X0.5', 'X0.1', 'X2', 'X10']]  # possible multipliers
        # Show message with option buttons
        update.message.reply_text(
            'Change learning rate, multiply by a factor of: '
            '(Send /cancel to leave LR unchanged).\n\n',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard))
        return 1

    def set_lr_back(self, bot, update):
        """ Telegram bot callback for the /setlr command. Handle user selection as part of conversation"""
        options = {'X0.5': 0.5, 'X0.1': 0.1, 'X2': 2.0, 'X10': 10.0}  # possible multipliers
        self.modify_lr = options[update.message.text]  # User selection
        update.message.reply_text(" Learning rate will be multiplied by {} on the beginning of next epoch!"
                                  .format(str(self.modify_lr)), reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_lr(self, bot, update):
        """ Telegram bot callback for the /setlr command. Handle user cancellation as part of conversation"""
        self.modify_lr = 1.0
        update.message.reply_text('OK, learning rate will not be modified on next epoch.',
                                  reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def lr_handler(self):
        """ Function to setup the callbacks for the /setlr command. Returns a conversation handler """
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('setlr', self.set_lr_front, filters=self.filters)],
            states={1: [RegexHandler('^(X0.5|X0.1|X2|X10)$', self.set_lr_back)]},
            fallbacks=[CommandHandler('cancel', self.cancel_lr, filters=self.filters)])

        return conv_handler

    # Stop training process callbacks
    def stop_training(self, bot, update):
        """ Telegram bot callback for the /stoptraining command. Displays verification message with buttons"""
        reply_keyboard = [['Yes', 'No']]
        update.message.reply_text(
                    'Are you sure? '
                    'This will stop your training process!\n\n',
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard))
        return 1

    def stop_training_verify(self, bot, update):
        """ Telegram bot callback for the /stoptraining command. Handle user selection as part of conversation"""
        is_sure = update.message.text  # Get response
        if is_sure == 'Yes':
            self.stop_train_flag = True
            update.message.reply_text('OK, stopping training!', reply_markup=ReplyKeyboardRemove())
        elif is_sure == 'No':
            self.stop_train_flag = False  # to allow changing your mind before stop took place
            update.message.reply_text('OK, canceling stop request!', reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_stop(self, bot, update):
        """ Telegram bot callback for the /stoptraining command. Handle user cancellation as part of conversation"""
        self.stop_train_flag = False
        update.message.reply_text('OK, training will not be stopped.',
                                  reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    def stop_handler(self):
        """ Function to setup the callbacks for the /stoptraining command. Returns a conversation handler """
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('stoptraining', self.stop_training, filters=self.filters)],
            states={1: [RegexHandler('^(Yes|No)$', self.stop_training_verify)]},
            fallbacks=[CommandHandler('cancel', self.cancel_stop, filters=self.filters)])
        return conv_handler

    # Plot loss history
    def plot_loss(self, bot, update):
        """ Telegram bot callback for the /plot command. Replies with a convergence plot image"""

        if not self.loss_hist or plt is None:
            # First epoch wasn't finished or matplotlib isn't installed
            return
        loss_np = np.asarray(self.loss_hist)
        # Check if training has a validation set
        val_loss_np = np.asarray(self.val_loss_hist) if self.val_loss_hist else None
        legend_keys = ['loss', 'val_loss'] if self.val_loss_hist else ['loss']

        x = np.arange(len(loss_np))  # Epoch axes
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, loss_np, 'b')  # Plot training loss
        if val_loss_np is not None:
            ax.plot(x, val_loss_np, 'r')  # Plot val loss
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax.legend(legend_keys)
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        update.message.reply_photo(buffer)  # Sent image to user


class TelegramBotCallback(Callback):
    """Callback that sends metrics and responds to Telegram Bot.
    Supports the following commands:
     /start: activate automatic updates every epoch
     /help: get a reply with all command options
     /status: get a reply with the latest epoch's results
     /getlr: get a reply with the current learning rate
     /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
     /plot: get a reply with the loss convergence plot image
     /quiet: stop getting automatic updates each epoch
     /stoptraining: kill Keras training process

    # Arguments
        kbot: Instance of the DLBot class, holding the appropriate bot token

    # Raises
        TypeError: In case kbot is not a DLBot instance.
    """

    def __init__(self, kbot):
        super(TelegramBotCallback, self).__init__()
        if isinstance(kbot, DLBot):
            self.kbot = kbot
        else:
            raise TypeError('Bot must be an instance of the DLBot class')

    def on_train_begin(self, logs=None):
        logs['lr'] = K.get_value(self.model.optimizer.lr)  # Add learning rate to logs dictionary
        self.kbot.lr = logs['lr']  # Update bot's value of current LR
        self.kbot.activate_bot()  # Activate the telegram bot
        self.epochs = self.params['epochs']  # number of epochs
        # loss history tracking
        self.loss_hist = []
        self.val_loss_hist = []

    def on_train_end(self, logs=None):
        self.kbot.send_message('Train Completed!')
        self.kbot.stop_bot()

    def on_epoch_begin(self, epoch, logs=None):
        # Check if learning rate should be changed
        if self.kbot.modify_lr != 1:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = float(K.get_value(self.model.optimizer.lr))  # get current lr
            # new LR
            lr = lr*self.kbot.modify_lr
            K.set_value(self.model.optimizer.lr, lr)
            self.kbot.modify_lr = 1  # Set multiplier back to 1

            message = '\nEpoch %05d: setting learning rate to %s.' % (epoch + 1, lr)
            print(message)
            self.kbot.send_message(message)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Did user invoke STOP command
        if self.kbot.stop_train_flag:
            self.model.stop_training = True
            self.kbot.send_message('Training Stopped!')
            print('Training Stopped! Stop command sent via Telegram bot.')

        # LR handling
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.kbot.lr = logs['lr']  # Update bot's value of current LR

        # Epoch message handling
        tlogs = ', '.join([k+': '+'{:.4f}'.format(v) for k, v in zip(logs.keys(), logs.values())])  # Clean logs string
        message = 'Epoch %d/%d \n' % (epoch + 1, self.epochs) + tlogs
        # Send epoch end logs
        if self.kbot.verbose:
            self.kbot.send_message(message)
        # Update status message
        self.kbot.set_status(message)

        # Loss tracking
        # Track loss to export as an image
        self.loss_hist.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_loss_hist.append(logs['val_loss'])
        self.kbot.loss_hist = self.loss_hist
        self.kbot.val_loss_hist = self.val_loss_hist
