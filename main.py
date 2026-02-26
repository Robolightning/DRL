# -*- coding: utf-8 -*-
# main.py – скрипт для тестирования предобученной модели DRL на датасете map2019

from __future__ import print_function, division

from types import SimpleNamespace

from test_meter import (
    create_model,
    create_dataset,
    test
)

# ------------------------------------------------------------------------------
# Настраиваемые пути (измените при необходимости)
# Указываем путь к папке, которая содержит подпапки с последовательностями (Chuanmei_80_*)
DATA_ROOT = "./map2019/merge_test_700-1800_cr0.95_stride100"
CHECKPOINT_PATH = "./checkpoints/net_best.pth"  # путь к чекпоинту
FILTER_R = 31                                   # параметр фильтра (должен совпадать с обучением)
NUM_WORKERS = 8                                 # число воркеров для загрузки данных
BATCH_SIZE = 8                                  # размер батча
# ------------------------------------------------------------------------------

# Конфигурация модели (взято из файла MixCvT13_FPN128_outstride1nearest_CE_Balance_cr31.py)
model_config = dict(
    backbone=dict(
        type="MixFormer",
        vit_type="cvt13",
        pretrain_path="pretrain_model/CvT-13-384x384-IN-22k.pth",
        pretrain=True,
        output_index=[0, 1, 2],
    ),
    neck=dict(
        type="FPN_I3",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=0,
    ),
    head=dict(
        type="ChannelEmbedding", input_ndim=128, mid_process_channels=[64, 16, 1]
    ),
    postprocess=dict(
        upsample_to_original=True,
        upsample_method="NearstUpsample",
        output_size=[384, 384],
    ),
    loss=dict(
        cls_loss=dict(type="BalanceLoss", center_R=31, neg_weight=130),
    ),
)

# Конфигурация данных
data_config = dict(
    batchsize=BATCH_SIZE,
    num_worker=NUM_WORKERS,
    val_batchsize=BATCH_SIZE,
    train_dir="",                     # не используется
    val_dir="",                        # не используется
    test_dir=DATA_ROOT,                # корень тестовых данных (папка с подпапками последовательностей)
    test_mode="",                      # оставляем пустым, чтобы не добавлять дополнительный сегмент пути
    UAVhw=[128, 128],
    Satellitehw=[384, 384],
)

# pipeline (не используется напрямую в тесте, но может потребоваться для совместимости)
pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            RandomErasing=dict(probability=0.3),
            RandomResize=dict(img_size=data_config["UAVhw"]),
            ToTensor=dict(),
        ),
        Satellite=dict(
            RandomCrop=dict(cover_rate=0.85, map_size=(512, 1000)),
            RandomResize=dict(img_size=data_config["Satellitehw"]),
            ToTensor=dict(),
        ),
    ),
)

# Конфигурация тестирования
test_config = dict(
    num_worker=NUM_WORKERS,
    filterR=FILTER_R,
    checkpoint=CHECKPOINT_PATH,
)

# Остальные параметры (не влияют на тест, но нужны для совместимости)
lr_config = dict(lr=1.5e-4, type="cosine", warmup_iters=500, warmup_ratio=0.01)
train_config = dict(autocast=True, num_epochs=12)
checkpoint_config = dict(interval=1, epoch_start_save=6, only_save_best=True)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666

# Создаём объект opt, аналогичный тому, что формируется в get_opt() из test_meter.py
opt = SimpleNamespace()
opt.k = 10                                   # параметр K для RDS (из argparse)
opt.model = model_config
opt.data_config = data_config
opt.pipline_config = pipline_config
opt.lr_config = lr_config
opt.train_config = train_config
opt.test_config = test_config
opt.checkpoint_config = checkpoint_config
opt.log_interval = log_interval
opt.load_from = load_from
opt.resume_from = resume_from
opt.debug = debug
opt.seed = seed

# Добавляем поля, которые формируются в get_opt() динамически
opt.savename = "result_filterR-{}.txt".format(opt.test_config["filterR"])
opt.GPS_output_filename = "GPS_pred_gt_filterR-{}.json".format(opt.test_config["filterR"])


# ------------------------------------------------------------------------------
# Вспомогательные функции из test_meter.py (скопированы без изменений)
# ------------------------------------------------------------------------------


GPS_output_list = []   # глобальный список для сохранения результатов

# ------------------------------------------------------------------------------
# Основная часть
# ------------------------------------------------------------------------------
def main():
    print("Загружаем модель...")
    model = create_model(opt)
    print("Модель загружена. Подготавливаем данные...")
    dataloader = create_dataset(opt)
    print("Начинаем тестирование...")
    test(model, dataloader, opt)
    print("Тестирование завершено. Результаты сохранены в папке output/")


if __name__ == '__main__':
    main()